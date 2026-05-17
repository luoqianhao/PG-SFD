import torch
from torch.nn import functional as F
import numpy as np
import os
import argparse
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from utils_multiTask import global_cosine_hm_adaptive
from models.v3_dinomaly import MultiTaskDinomaly

label_mapping = {0: "good", 1: 'break_large', 2: 'break_small',3: "contamination"}
# label_mapping = {0: "normal", 1: 'mix', 2: 'neg', 3:'zg'}

def load_model(args, device):
    model = MultiTaskDinomaly(encoder_name=args.encoder, remove_class_token=True, inp_num=args.INP_num, num_classes=args.num_classes)
    model = model.to(device)

    # 加载权重
    print(f"save_dir: {args.model_dir}, save_name:{args.save_name}")
    model_path = os.path.join(args.model_dir, args.save_name, 'model_epoch_200_Spinal.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model

def preprocess_image(image_path, input_size=448, crop_size=392, crop_pix=100):
    # 和训练代码保持一样的数据预处理
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    img = Image.open(image_path).convert('RGB')
    if crop_pix > 0:
        w, h = img.size
        img = img.crop((0, crop_pix, w, h - crop_pix))  # (left, top, right, bottom)
    
    img_tensor = transform(img).unsqueeze(0)
    
    return img_tensor, img

def visualize_results(img_tensor, img_orig, anomaly_map, save_path=None, strongest_prototype_id=None, strongest_response=None):
    # Convert tensors to numpy arrays
    img_np = img_tensor.squeeze().permute(1, 2, 0).numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    img_np = img_np.astype(np.uint8)
    
    # Process anomaly map
    anomaly_map = anomaly_map.squeeze().cpu().numpy()
    print("anomaly_map max pre: ", anomaly_map.max())
    # anomaly_map[180:205,200:220] = anomaly_map.min()
    # print("anomaly_map max after: ", anomaly_map.max())
    anomaly_map_max = anomaly_map.max()
    if anomaly_map_max > 0.01: # 可能是异常值,否则zero
        anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    else:
        anomaly_map = anomaly_map #np.zeros_like(anomaly_map)
    # anomaly_map = anomaly_map.squeeze().cpu().numpy()
    # anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min()) * 255
    anomaly_map = (anomaly_map * 255).astype(np.uint8)

    anomaly_map_jet = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    anomaly_map_winter = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_WINTER)
    anomaly_map_jet = cv2.resize(anomaly_map_jet, (img_np.shape[1], img_np.shape[0]))
    anomaly_map_winter = cv2.resize(anomaly_map_winter, (img_np.shape[1], img_np.shape[0]))

    anomaly_map_show = anomaly_map_jet
    # anomaly_map_show = cv2.cvtColor(anomaly_map_show, cv2.COLOR_BGR2RGB)
    save_amp_path = os.path.dirname(save_path)
    save_amp_path = os.path.join(save_amp_path,f'{os.path.splitext(save_path)[0]}_amp.png')  #保存热力图
    print(save_amp_path)
    cv2.imwrite(save_amp_path, anomaly_map_show)

    img_rec_anomaly_map = cv2.addWeighted(img_np, 1, anomaly_map_show, 0.5, 0.0)
    cv2.imwrite(save_path, img_rec_anomaly_map)

    anomaly_map_winter_g_channel = anomaly_map_winter[:,:,1]

    _, mask = cv2.threshold(anomaly_map_winter_g_channel, 127, 255, cv2.THRESH_BINARY)
    # mask = 255 - mask
    
    # Create figure
    # plt.figure(figsize=(15, 5))
    
    # # Original image
    # plt.subplot(1, 3, 1)
    # plt.imshow(img_np)
    # plt.title('Original Image')
    # plt.axis('off')
    
    # # Anomaly map
    # plt.subplot(1, 3, 2)
    # plt.imshow(anomaly_map_show)
    # plt.title('Anomaly Map')
    # plt.axis('off')
    
    # # Overlay
    # plt.subplot(1, 3, 3)
    # plt.imshow(mask, cmap='gray')
    # plt.title('mask')
    # plt.axis('off')

    # Add prototype response info if available
    # if strongest_prototype_id is not None and strongest_response is not None:
    #     plt.suptitle(f'Strongest Prototype ID: {label_mapping[strongest_prototype_id]}, Response: {strongest_response:.4f}', fontsize=16)
    
    # # Save or show
    # if save_path:
    #     plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    #     plt.close()
    # else:
    #     plt.tight_layout()
    #     plt.show()

def test_global_cosine_hm_percent(fs_list, ft_list, out_size=224):
    # cos_loss = torch.nn.CosineSimilarity()
    # return 1-cos_loss(fs_list[-1], ft_list[-1]).unsqueeze(1), out_size

    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = F.cosine_similarity(fs, ft)
        print(a_map.shape)
        # mse_map = torch.mean((fs-ft)**2, dim=1)
        # a_map = mse_map
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = 1 - a_map
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list

def process_single_image(model, image_path, device, save_dir=None):
    # Preprocess image
    img_tensor, img_orig = preprocess_image(image_path, args.input_size, args.crop_size, crop_pix=0)
    img_tensor = img_tensor.to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(img_tensor)
        en, de = outputs['anomaly_features']
        g_loss = outputs['gather_loss']
        cls_logits = outputs['cls_logits']
        
        # 1. 输出异常结果
        anomaly_map, _ = test_global_cosine_hm_percent(en, de)
        global_score = g_loss.item()  # 或者其他图像级得分
        print(anomaly_map.shape)
        print(f"Global score : {global_score:.4f}")
        # anomaly_map = anomaly_map.squeeze().cpu().numpy()

        # 2. 输出分类结果
        # cls_pred = torch.argmax(cls_logits, dim=-1)
        # strongest_prototype_id, strongest_response = cls_pred.item(), cls_logits[0][cls_pred].item()
        # label_pred = label_mapping[strongest_prototype_id]
        # print(f"Strongest responding prototype ID: {label_pred} with score {strongest_response}")
        
    
    # Determine save path if needed
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(image_path))[0] + '_result.jpg'
        save_path = os.path.join(save_dir, filename)
        print(save_path)
    
    # Visualize results
    # visualize_results(img_tensor.cpu(), img_orig, anomaly_map.squeeze(), save_path,strongest_prototype_id,strongest_response)
    visualize_results(img_tensor.cpu(), img_orig, anomaly_map.squeeze(), save_path)
def process_folder(model, folder_path, device, save_dir=None):
    # Get all image files in folder
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(valid_extensions)]
    
    # Process each image
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        process_single_image(model, img_path, device, save_dir)
        print(f"Processed: {img_file}")
        # try:
        #     process_single_image(model, img_path, device, save_dir)
        #     print(f"Processed: {img_file}")
        # except Exception as e:
        #     print(f"Error processing {img_file}: {str(e)}")

if __name__ == '__main__':
    # Parse arguments datasets/mvtec_anomaly_detection/bottle/test/contamination/000.png
    parser = argparse.ArgumentParser(description='INP-Former Visualization')
    parser.add_argument('--input_path', type=str, default= "/data/ymt/INP-Former-main/test_images/vis/Spinal", 
                       help='Path to input image or folder')
    parser.add_argument('--save_dir', type=str, default='/data/ymt/INP-Former-main/test_images/vis/Spinal/res_ours',
                       help='Directory to save results (optional)')
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_small_14',
                       help='Encoder architecture')
    parser.add_argument('--input_size', type=int, default=448,
                       help='Input image size')
    parser.add_argument('--crop_size', type=int, default=448,
                       help='Cropped image size')
    parser.add_argument('--INP_num', type=int, default=6,
                       help='Number of INP tokens')
    parser.add_argument('--save_name', type=str, default='mutiTaskv3Dinomaly_Medical_Spinal_cls2.0_noCrop-class-20251022_ymt',
                       help='Name of saved model directory')
    parser.add_argument('--model_dir', type=str, default='./saved_results/',
                       help='Directory containing saved models')
    parser.add_argument('--num_classes', type=int, default=3)
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = load_model(args, device)
    
    # Determine if input is a single image or folder
    if os.path.isfile(args.input_path):
        process_single_image(model, args.input_path, device, args.save_dir)
    elif os.path.isdir(args.input_path):
        process_folder(model, args.input_path, device, args.save_dir)
    else:
        raise ValueError("Input path must be a valid file or directory")