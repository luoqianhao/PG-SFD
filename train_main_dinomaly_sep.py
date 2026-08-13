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
from models.v3_dinomaly import MultiTaskDinomaly

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
    
def main(args,item):
    # Fixing the Random Seed
    setup_seed(1)

    # Data Preparation
    data_transform, gt_transform = get_data_transforms(args.input_size, args.crop_size)

    if args.dataset == 'MVTec-AD' or args.dataset == 'VisA' or args.dataset == 'Medical_data':
        train_data_list = []
        test_data_list = []
        # Build the global class map.
        all_defects = set()
    
        train_dir = os.path.join(args.data_path, item, "train")
        test_dir = os.path.join(args.data_path, item, "test")
        for d in os.listdir(train_dir):
            if d != "good" and os.path.isdir(os.path.join(train_dir, d)):
                all_defects.add(d)
        for d in os.listdir(test_dir):
            if d != "good" and os.path.isdir(os.path.join(test_dir, d)):
                all_defects.add(d)

        num_classes_total = 1 + len(all_defects)   # 0=good，1..=defects
        args.num_classes = num_classes_total
        
        global_cls_map = {n: (i+1) for i, n in enumerate(sorted(all_defects))}
        print_fn(global_cls_map)

        train_path = os.path.join(args.data_path, item)
        test_path = os.path.join(args.data_path, item)

        train_data = MVTecDataset_v2(root=train_path, transform=data_transform, gt_transform=gt_transform,phase='train', mode='train_with_anomalies', global_cls_map=global_cls_map)
        cls_map = train_data.get_class_mapping()
        train_data_list.append(train_data)
        

        test_data = MVTecDataset_v2(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test", global_cls_map=global_cls_map)
        test_data_list.append(test_data)
            
        train_data = ConcatDataset(train_data_list)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    model = MultiTaskDinomaly(
        encoder_name=args.encoder,
        remove_class_token=True,
        inp_num=args.INP_num,
        num_classes=args.num_classes,
        enable_eq67=args.enable_eq67,
        separation_tau=args.separation_tau,
    )
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
            separation_loss_list = []
            consistency_loss_list = []
            total_loss_list = []
            # for img, labels in tqdm(train_dataloader, ncols=80):  # Labels are required.
            for img, gt, labels, img_path in tqdm(train_dataloader, ncols=80):
                # print(f"labels:{labels}")  # Debug loaded labels.
                img = img.to(device)
                labels = labels.to(device)
            
                eq67_active = (
                    args.enable_eq67 and epoch >= args.eq67_start_epoch
                )
                outputs = model(img, compute_eq67=eq67_active)
                en, de = outputs['anomaly_features']
                g_loss = outputs['gather_loss']
                cls_logits = outputs['cls_logits']
                anomaly_prototypes = outputs['anomaly_prototypes']        # [B, K, D]
                class_prototypes = outputs['class_prototypes']            # [B，C, D]
                separation_loss = outputs['separation_loss']
                consistency_loss = outputs['consistency_loss']

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

                # Eq. (6)/(7) are additive and do not replace the released
                # gather/prototype-alignment terms. With the default start=60,
                # they become active at displayed epoch 61.
                if eq67_active:
                    loss = loss + \
                        args.lambda_sep * separation_loss + \
                        args.lambda_cons * consistency_loss

                # Backpropagate.
                optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()

                # Record losses.
                loss_list.append(loss.item())
                anomaly_loss_list.append(anomaly_loss.item())
                cls_loss_list.append(cls_loss.item())
                separation_loss_list.append(separation_loss.item())
                consistency_loss_list.append(consistency_loss.item())
                total_loss_list.append(loss.item())
                
                lr_scheduler.step()
        
            # Record and report the mean epoch losses.
            print_fn('epoch [{}/{}], total_loss:{:.4f}, anomaly_loss:{:.4f}, cls_loss:{:.4f}, Lsep:{:.4f}, Lcons:{:.4f}'.format(
                epoch+1, args.total_epochs, 
                np.mean(total_loss_list), 
                np.mean(anomaly_loss_list),
                np.mean(cls_loss_list),
                np.mean(separation_loss_list),
                np.mean(consistency_loss_list)
            ))
            if (epoch + 1) % 50 == 0 or (epoch + 1) == args.total_epochs:
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
                
                for test_data in test_data_list:
                    test_dataloader = torch.utils.data.DataLoader(
                        test_data, batch_size=args.batch_size, shuffle=False, num_workers=4
                    )
                    
                    # Evaluate all tasks.
                    results = evaluation_batch_simple(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, cls_acc, cls_f1 = results
                     
                    print_fn('{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}, Cls-Acc:{:.4f}, Cls-F1:{:.4f}'.format(
                        item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, cls_acc, cls_f1
                    ))

                torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, f'model_epoch_{epoch+1}_{item}.pth'))
                model.train()
        return auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, cls_acc, cls_f1
                
    elif args.phase == 'test':
        # Test
        checkpoint_path = args.checkpoint or os.path.join(
            args.save_dir, args.save_name, 'model.pth'
        )
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device), strict=True
        )
        auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
        auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
        cls_acc_list, cls_f1_list = [], []
        model.eval()
        for test_data in  test_data_list:
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
        return (
            np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
            np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list),
            np.mean(aupro_px_list), np.mean(cls_acc_list), np.mean(cls_f1_list),
        )


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "3"
    parser = argparse.ArgumentParser(description='')

    # dataset info
    parser.add_argument('--dataset', type=str, default=r'MVTec-AD') # 'MVTec-AD' or 'VisA' or 'Real-IAD' or 'Medical_data'
    parser.add_argument('--data_path', type=str, default=r'/home/cisdi/luoqianhao/dataset/anomaly_dataset/mvtec_2d/') # Replace it with your path.
    
    # save info
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default='mutiTaskv3Dinomaly_gangguan-20251108_lqh')
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='explicit checkpoint path for --phase test',
    )

    # model info
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_small_14') # 'dinov3_vits16'，'dinov2reg_vit_small_14' or 'dinov2reg_vit_base_14' or 'dinov2reg_vit_large_14'
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=448)
    parser.add_argument('--INP_num', type=int, default=6)
    parser.add_argument('--num_classes', type=int, default=3) # only for multi-class dataset

    # training info
    parser.add_argument('--total_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument(
        '--device', type=str,
        default='cuda:0' if torch.cuda.is_available() else 'cpu',
    )
    parser.add_argument(
        '--items', nargs='+', default=None,
        help='optional dataset categories; overrides the repository defaults',
    )

    # Eq. (6)/(7). These defaults are explicit implementation assumptions;
    # the paper does not publish lambda_sep/lambda_cons, the exact Eq. (6)
    # interpretation, or phi's architecture.
    eq67_group = parser.add_mutually_exclusive_group()
    eq67_group.add_argument(
        '--enable-eq67', dest='enable_eq67', action='store_true'
    )
    eq67_group.add_argument(
        '--disable-eq67', dest='enable_eq67', action='store_false'
    )
    parser.set_defaults(enable_eq67=True)
    parser.add_argument('--lambda-sep', type=float, default=0.2)
    parser.add_argument('--lambda-cons', type=float, default=0.1)
    parser.add_argument('--separation-tau', type=float, default=1.0)
    parser.add_argument(
        '--eq67-start-epoch', type=int, default=60,
        help='zero-based activation epoch; 60 means displayed epoch 61',
    )

    args = parser.parse_args()
    if args.lambda_sep < 0 or args.lambda_cons < 0:
        parser.error('Eq. (6)/(7) weights must be non-negative')
    if not 0 <= args.eq67_start_epoch < args.total_epochs:
        parser.error('--eq67-start-epoch must be in [0, total_epochs)')
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    print_fn(
        'Geometry-regularized Optimization config: enabled={}, lambda_sep={}, lambda_cons={}, tau={}, '
        'zero_based_start_epoch={} (displayed epoch {})'.format(
            args.enable_eq67,
            args.lambda_sep,
            args.lambda_cons,
            args.separation_tau,
            args.eq67_start_epoch,
            args.eq67_start_epoch + 1,
        )
    )
    device = args.device

    # category info
    if args.items is not None:
        args.item_list = args.items
    elif args.dataset == 'MVTec-AD':
        # args.item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule',
        #                 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
        args.item_list = ['gangguan_2d_multiTask_1208']
    elif args.dataset == 'VisA':
        # args.data_path = r'E:\IMSN-LW\dataset\VisA_pytorch\1cls'  # '/path/to/dataset/VisA/'
        args.item_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif args.dataset == 'Medical_data':
        args.item_list = ['brain', 'liver', 'retinal']
        # args.item_list = ['Spinal']
        
    result_list = []
    for i, item in enumerate(args.item_list):

        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, cls_acc, cls_f1 = main(args, item)
        result_list.append([item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px,cls_acc,cls_f1])

    mean_auroc_sp = np.mean([result[1] for result in result_list])
    mean_ap_sp = np.mean([result[2] for result in result_list])
    mean_f1_sp = np.mean([result[3] for result in result_list])

    mean_auroc_px = np.mean([result[4] for result in result_list])
    mean_ap_px = np.mean([result[5] for result in result_list])
    mean_f1_px = np.mean([result[6] for result in result_list])
    mean_aupro_px = np.mean([result[7] for result in result_list])

    mea_cls_acc = np.mean([result[8] for result in result_list])
    mea_cls_f1 = np.mean([result[9] for result in result_list])

    print_fn(result_list)
    print_fn(
        'Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}, Cls-Acc:{:.4f}, Cls-F1:{:.4f}'.format(
            mean_auroc_sp, mean_ap_sp, mean_f1_sp,
            mean_auroc_px, mean_ap_px, mean_f1_px, mean_aupro_px,
            mea_cls_acc,mea_cls_f1))
