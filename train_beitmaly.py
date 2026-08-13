import torch
import torch.nn as nn
import numpy as np
import os
from functools import partial
import warnings
from tqdm import tqdm
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
import argparse
from optimizers import StableAdamW
from utils_multiTask import evaluation_batch,WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger, evaluation_batch_simple

# Dataset-Related Modules
from dataset import MVTecDataset, MVTecTrainWithAnomalies, MVTecDataset_v2
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

# Model-Related Modules
from models.ibot_dinomaly import MultiTaskIBOTmaly
from models.beit_dinomaly import MultiTaskBeitmaly

# Ignore module-specific FutureWarning messages.
warnings.filterwarnings("ignore", category=FutureWarning, module="kornia.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.*")
warnings.filterwarnings("ignore")

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
    
def main(args):
    # Fixing the Random Seed
    setup_seed(1)

    # Data Preparation
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    if args.dataset == 'MVTec-AD' or args.dataset == 'VisA':
        train_data_list = []
        test_data_list = []
        # Build the global class map.
        all_defects = set()
        for item in args.item_list:
            train_dir = os.path.join(args.data_path, item, "train")
            test_dir = os.path.join(args.data_path, item, "test")
            for d in os.listdir(train_dir):
                if d != "good" and os.path.isdir(os.path.join(train_dir, d)):
                    all_defects.add(d)
            for d in os.listdir(test_dir):
                if d != "good" and os.path.isdir(os.path.join(test_dir, d)):
                    all_defects.add(d)

        global_cls_map = {n: (i+1) for i, n in enumerate(sorted(all_defects))}
        print_fn(global_cls_map)
        for i, item in enumerate(args.item_list):
            train_path = os.path.join(args.data_path, item)
            test_path = os.path.join(args.data_path, item)

            train_data = MVTecDataset_v2(root=train_path, transform=data_transform, gt_transform=gt_transform,phase='train', mode='train_with_anomalies', global_cls_map=global_cls_map)
            cls_map = train_data.get_class_mapping()
            train_data_list.append(train_data)
            
            test_data = MVTecDataset_v2(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test", global_cls_map=global_cls_map)
            test_data_list.append(test_data)
            
        train_data = ConcatDataset(train_data_list)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    model = MultiTaskBeitmaly(encoder_name=args.encoder, remove_class_token=True, inp_num=args.INP_num, num_classes=args.num_classes)
    model = model.to(device)

    if args.phase == 'train':
        # Initialize parameters.
        for module in [model.anomaly_module, model.classification_module]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
        # Initialize prototype tokens separately.
        with torch.no_grad():
            # Initialize anomaly prototypes.
            trunc_normal_(model.anomaly_module.anomaly_prototypes, std=0.01, a=-0.03, b=0.03)
            # Initialize class prototypes.
            trunc_normal_(model.classification_module.class_prototypes, std=0.01, a=-0.03, b=0.03)
        
        # Use task-specific learning rates.
        optimizer = StableAdamW([
            {'params': model.anomaly_module.parameters(), 'lr': 1e-3, 'name': 'anomaly'},
            {'params': model.classification_module.parameters(), 'lr': 5e-3, 'name': 'classification'}
        ], betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
        
        lr_scheduler = WarmCosineScheduler(
            optimizer, base_value=1e-3, final_value=1e-4, 
            total_iters=args.total_epochs*len(train_dataloader),
            warmup_iters=100
        )
    
        print_fn('train image number:{}'.format(len(train_data)))

        # Train
        for epoch in range(args.total_epochs):
            model.train()
            # Add monitoring metrics.
            loss_list = []
            anomaly_loss_list = []
            cls_loss_list = []
            total_loss_list = []
            # for img, labels in tqdm(train_dataloader, ncols=80):  # Labels are required.
            for img, gt, labels, img_path in tqdm(train_dataloader, ncols=80):
                # print(f"labels:{labels}")  # Debug loaded labels.
                img = img.to(device)
                labels = labels.to(device)
            
                outputs = model(img)
                en, de = outputs['anomaly_features']
                g_loss = outputs['gather_loss']
                cls_logits = outputs['cls_logits']
                anomaly_prototypes = outputs['anomaly_prototypes']        # [B, K, D]
                class_prototypes = outputs['class_prototypes']            # [B，C, D]

                # Compute losses.
                # 1. Reconstruction loss on normal samples only.
                 # anomaly_loss = global_cosine_hm_adaptive(en, de, y=3)
                normal_mask = (labels == 0)  # True marks normal samples.
                if normal_mask.any():  # At least one normal sample.
                    en_normal = [e[normal_mask] for e in en]      # List[Tensor]
                    de_normal = [d[normal_mask] for d in de]      # List[Tensor]
                    anomaly_loss = global_cosine_hm_adaptive(en_normal, de_normal, y=3)
                else:
                    anomaly_loss = torch.tensor(0.0, device=img.device, requires_grad=False) # Skip when no normal samples exist.
                
                # 2. Classification loss.
                # cls_loss = F.cross_entropy(cls_logits, labels)
                cls_loss = FocalLoss(gamma=2)(cls_logits, labels)
                
                # 3. Prototype alignment loss.
                # Select anomaly prototypes for normal samples.
                anomaly_proto_normal = anomaly_prototypes[normal_mask]  # [B_n, K, D]
                # Select the normal-class prototype.
                class_proto_normal = class_prototypes[:, 0:1, :]         # [B, 1, D], shared by all samples.
                if anomaly_proto_normal.size(0) > 0:
                    # Align each normal anomaly prototype with class_prototypes[0].
                    # Use the mean or minimum distance.
                    dist = F.mse_loss(
                        anomaly_proto_normal.mean(1),  # [B_n, D]
                        class_proto_normal[0].expand(anomaly_proto_normal.size(0), -1)  # [B_n, D]
                    )
                    proto_align_loss = dist
                else:
                    proto_align_loss = torch.tensor(0.0, device=img.device, requires_grad=False) # Skip when no normal samples exist.

                
                # Combine losses by curriculum stage.
                # Stage 1: classification only.
                if epoch < 5:
                    loss = cls_loss

                # Stage 2: classification and anomaly detection.
                elif epoch < 60:
                    # Upweight anomaly_loss relative to cls_loss.
                    loss = 1.0 * cls_loss + 2.0 * anomaly_loss  # Emphasize anomaly detection.

                # Stage 3: add alignment loss from epoch 60.
                else:
                    loss = 2.0 * cls_loss + 1.0 * anomaly_loss + \
                        0.2 * g_loss + 0.1 * proto_align_loss

                # Backpropagate.
                optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()

                # Record losses.
                loss_list.append(loss.item())
                anomaly_loss_list.append(anomaly_loss.item())
                cls_loss_list.append(cls_loss.item())
                total_loss_list.append(loss.item())
                
                lr_scheduler.step()
        
            # Record and report the mean epoch losses.
            print_fn('epoch [{}/{}], total_loss:{:.4f}, anomaly_loss:{:.4f}, cls_loss:{:.4f}'.format(
                epoch+1, args.total_epochs, 
                np.mean(total_loss_list), 
                np.mean(anomaly_loss_list),
                np.mean(cls_loss_list)
            ))
            if (epoch + 1) % 50 == 0 or (epoch + 1) == args.total_epochs:
                # print("\n--- Gradient Check ---")
                # for name, param in model.classification_module.named_parameters():
                #     if param.grad is not None:
                #         print(f"{name}: {param.grad.norm().item():.6f}")
                print(f"Labels: {labels[:5].cpu().numpy()}, unique: {labels.unique().cpu().numpy()}")
                cls_pred = torch.argmax(cls_logits, dim=-1)
                print(f"Preds:  {cls_pred[:5].cpu().numpy()}, bincount: {cls_pred.bincount().cpu().numpy()}")
                print(f"CE Loss: {cls_loss.item():.4f}, Logits mean/std: {cls_logits.mean().item():.4f}/{cls_logits.std().item():.4f}")
                acc = (cls_pred == labels).float().mean().item()
                print(f"Logits: {cls_logits[0].detach()}, Acc: {acc:.3f}")

                model.eval()
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
                cls_acc_list, cls_f1_list = [], []
                
                for item, test_data in zip(args.item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(
                        test_data, batch_size=args.batch_size, shuffle=False, num_workers=4
                    )
                    
                    # Evaluate all tasks.
                    results = evaluation_batch_simple(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, cls_acc, cls_f1 = results
                    
                    # Record metrics.
                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)
                    cls_acc_list.append(cls_acc)
                    cls_f1_list.append(cls_f1)
                    
                    print_fn('{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}, Cls-Acc:{:.4f}, Cls-F1:{:.4f}'.format(
                        item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, cls_acc, cls_f1
                    ))
                
                # Report mean metrics.
                print_fn('Mean - I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}, Cls-Acc:{:.4f}, Cls-F1:{:.4f}'.format(
                    np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                    np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), 
                    np.mean(aupro_px_list), np.mean(cls_acc_list), np.mean(cls_f1_list)
                ))
                
                # Save the model.
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, f'model_epoch_{epoch+1}.pth'))
                model.train()
    elif args.phase == 'test':
        # Test
        model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_name, 'model.pth')), strict=True)
        auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
        auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
        cls_acc_list, cls_f1_list = [], []
        model.eval()
        for item, test_data in zip(args.item_list, test_data_list):
            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=4)
            # Evaluate all tasks.
            results = evaluation_batch_simple(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
            auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, cls_acc, cls_f1 = results
            
            # Record metrics.
            auroc_sp_list.append(auroc_sp)
            ap_sp_list.append(ap_sp)
            f1_sp_list.append(f1_sp)
            auroc_px_list.append(auroc_px)
            ap_px_list.append(ap_px)
            f1_px_list.append(f1_px)
            aupro_px_list.append(aupro_px)
            cls_acc_list.append(cls_acc)
            cls_f1_list.append(cls_f1)
            
            print_fn('{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}, Cls-Acc:{:.4f}, Cls-F1:{:.4f}'.format(
                item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, cls_acc, cls_f1
            ))
        
        # Report mean metrics.
        print_fn('Mean - I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}, Cls-Acc:{:.4f}, Cls-F1:{:.4f}'.format(
            np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
            np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), 
            np.mean(aupro_px_list), np.mean(cls_acc_list), np.mean(cls_f1_list)
        ))


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "4"
    parser = argparse.ArgumentParser(description='')

    # dataset info
    parser.add_argument('--dataset', type=str, default=r'MVTec-AD') # 'MVTec-AD' or 'VisA' or 'Real-IAD'
    parser.add_argument('--data_path', type=str, default=r'/data/luoqianhao/datasets/anomaly_datesets/mvtec_2d/') # Replace it with your path.

    # save info
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='mutiTaskBeitmaly_bottle_cls2.0_noCrop-class-20250926-pr50_ymt')

    # model info
    parser.add_argument('--encoder', type=str, default='beit_vit_base') #'ibot_vit_small''mae_vit_base' 'dinov3_vits16' 'dinov2reg_vit_small_14' or 'dinov2reg_vit_base_14' or 'dinov2reg_vit_large_14'
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=448)
    parser.add_argument('--INP_num', type=int, default=6)
    parser.add_argument('--num_classes', type=int, default=4) # only for multi-class dataset

    # training info
    parser.add_argument('--total_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--phase', type=str, default='train')

    args = parser.parse_args()
    # args.save_name = args.save_name + f'_dataset={args.dataset}_Encoder={args.encoder}_Resize={args.input_size}_Crop={args.crop_size}_INP_num={args.INP_num}'
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    device = 'cuda:4' if torch.cuda.is_available() else 'cpu'

    # category info
    if args.dataset == 'MVTec-AD':
        # args.item_list = ['bottle','screw','grid','zipper']
        # args.item_list = ['gangguan_temp_cls_0902']
        args.item_list = ['bottle-cls']
    elif args.dataset == 'VisA':
        # args.data_path = r'E:\IMSN-LW\dataset\VisA_pytorch\1cls'  # '/path/to/dataset/VisA/'
        args.item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif args.dataset == 'Real-IAD':
        # args.data_path = 'E:\IMSN-LW\dataset\Real-IAD'  # '/path/to/dataset/Real-IAD/'
        args.item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']
    main(args)
