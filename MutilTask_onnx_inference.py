import torch
from torch.nn import functional as F
from time import time
import numpy as np
import os
import argparse
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import onnxruntime as ort
from utils_multiTask import global_cosine_hm_adaptive
# from models.my_model import MultiTaskINPFormer

# label_mapping = {0: "normal", 1: 'mix', 2: 'neg', 3:'zg'}
label_mapping = {0: "normal", 1: 'neg', 2:'zg'}
# label_mapping = {0: "good", 1: 'break_large', 2: 'break_small',3: "contamination"}

def load_onnx_session(args):
    providers = ['CUDAExecutionProvider']
    sess = ort.InferenceSession(args.onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    print(f"ONNX model loaded with input: {input_name}, outputs: {output_names}")
    return sess, input_name, output_names

def preprocess_image(image_path, input_size=448, crop_size=448):
    # 和训练代码保持一样的数据预处理
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    return img_tensor, img

def visualize_results(img_tensor, img_orig, anomaly_map, save_path=None, strongest_prototype_id=None, strongest_response=None):
    # Convert tensors to numpy arrays
    img_np = img_tensor.squeeze().transpose(1, 2, 0)
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    img_np = img_np.astype(np.uint8)
    
    # Process anomaly map
    anomaly_map = anomaly_map.squeeze().cpu().numpy()
    # print("anomaly_map max pre: ", anomaly_map.max())
    anomaly_map_max = anomaly_map.max()
    # anomaly_map[180:205,200:220] = anomaly_map.min()
    print("anomaly_map max:", anomaly_map.max())
    # anomaly_map_max = anomaly_map.max()
    if anomaly_map_max > 0.04: # 可能是异常值,否则zero
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
    anomaly_map_show = cv2.cvtColor(anomaly_map_show, cv2.COLOR_BGR2RGB)

    anomaly_map_winter_g_channel = anomaly_map_winter[:,:,1]

    _, mask = cv2.threshold(anomaly_map_winter_g_channel, 127, 255, cv2.THRESH_BINARY)
    
    # Create figure
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    
    # Anomaly map
    plt.subplot(1, 3, 2)
    plt.imshow(anomaly_map_show)
    plt.title('Anomaly Map')
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap="gray")
    plt.title('mask')
    plt.axis('off')

    # Add prototype response info if available
    if strongest_prototype_id is not None and strongest_response is not None:
        plt.suptitle(f'Strongest Prototype ID: {label_mapping[strongest_prototype_id]}, Response: {strongest_response:.4f}', fontsize=16)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
def test_global_cosine_hm_percent_enonly(fs_list, out_size=224):
    cos_loss = torch.nn.CosineSimilarity()
    return 1-cos_loss(fs_list[0], fs_list[1]).unsqueeze(1), out_size

def test_global_cosine_hm_percent(fs_list, ft_list, out_size=224):
    # cos_loss = torch.nn.CosineSimilarity()
    # return cos_loss(fs_list[0], ft_list[0]).unsqueeze(1), out_size

    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = F.cosine_similarity(fs, ft)
        # mse_map = torch.mean((fs-ft)**2, dim=1)
        # a_map = mse_map
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = 1 - a_map
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)
    return anomaly_map, a_map_list

def process_single_image(args, image_path, sess, input_name, output_names):
    # Preprocess image
    start_time = time()
    img_tensor, img_orig = preprocess_image(image_path, args.input_size, args.crop_size)
    img_tensor = img_tensor.cpu().numpy().astype(np.float32)

    outputs = sess.run(output_names, {input_name: img_tensor})
    end_time = time()
    en_list = outputs[0:2]
    de_list = outputs[2:4]
    _, _,_,_, g_loss_np, cls_logits_np, cls_sims_np, an_proto_np, cls_proto_np = outputs
    en_list = [torch.from_numpy(en_np).to(torch.float32) for en_np in en_list]   #en to tensor
    de_list = [torch.from_numpy(de_np).to(torch.float32) for de_np in de_list]   #de to tensor

    g_loss = torch.from_numpy(g_loss_np).to(torch.float32)   #gather_loss
    cls_logits = torch.from_numpy(cls_logits_np).to(torch.float32)   #cls_logits
    
    # 1. 输出异常结果
    anomaly_map, _ = test_global_cosine_hm_percent(en_list, de_list)
    # anomaly_map, _ = test_global_cosine_hm_percent_enonly(en_list)
    global_score = g_loss.item()  # 或者其他图像级得分
    print(anomaly_map.shape)
    print(f"Global score : {global_score:.4f}")

    # 2. 输出分类结果
    cls_pred = torch.argmax(cls_logits, dim=-1)
    strongest_prototype_id, strongest_response = cls_pred.item(), cls_logits[0][cls_pred].item()
    label_pred = label_mapping[strongest_prototype_id]
    print(f"Strongest responding prototype ID: {label_pred} with score {strongest_response}")
        
    # Determine save path if needed
    save_path = None
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(image_path))[0] + '_result.jpg'
        save_path = os.path.join(args.save_dir, filename)
        print(save_path)

    # Visualize results
    visualize_results(img_tensor, img_orig, anomaly_map.squeeze(), save_path,strongest_prototype_id,strongest_response)
    end_time_1 = time()
    print("inference time :", end_time - start_time)
    print("inference time + save time :", end_time_1 - start_time)

def process_folder(args, sess, input_name, output_names):
    # Get all image files in folder

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(args.input_path) 
                  if f.lower().endswith(valid_extensions)]
    
    # Process each image
    for img_file in image_files:
        img_path = os.path.join(args.input_path, img_file)
        process_single_image(args, img_path, sess, input_name, output_names)
        print(f"Processed: {img_file}")
        # try:
        #     process_single_image(model, img_path, device, save_dir)
        #     print(f"Processed: {img_file}")
        # except Exception as e:
        #     print(f"Error processing {img_file}: {str(e)}")

if __name__ == '__main__':
    # Parse arguments datasets/mvtec_anomaly_detection/bottle/test/contamination/000.png
    parser = argparse.ArgumentParser(description='INP-Former Visualization')
    parser.add_argument('--input_path', type=str, default="/home/cisdi/luoqianhao/dataset/anomaly_dataset/mvtec_2d/gangguan_2d_multiTask_1208/test/neg/2025-11-05-19-00-23_37.png", 
                       help='Path to input image or folder')
    parser.add_argument('--save_dir', type=str, default='/home/cisdi/luoqianhao/PSG-FSD/infer_result',
                       help='Directory to save results (optional)')
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_small_14',
                       help='Encoder architecture')
    parser.add_argument('--input_size', type=int, default=448,
                       help='Input image size')
    parser.add_argument('--crop_size', type=int, default=448,
                       help='Cropped image size')
    parser.add_argument('--INP_num', type=int, default=6,
                       help='Number of INP tokens')
    parser.add_argument('--num_prototypes', type=int, default=4,
                       help='Number of semantic prototypes')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--onnx_path', type=str, default="/home/cisdi/luoqianhao/PSG-FSD/saved_results/mutiTaskv3Dinomaly_gangguan-20251108_lqh/mutitaskv3dinomaly_1208.onnx",
                       help='Directory containing saved onnx')
    
    args = parser.parse_args()
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #加载onnx session
    sess, input_name, output_names = load_onnx_session(args)

    # Determine if input is a single image or folder
    if os.path.isfile(args.input_path):
        process_single_image(args, args.input_path, sess, input_name, output_names)
    elif os.path.isdir(args.input_path):
        process_folder(args, sess, input_name, output_names )
    else:
        raise ValueError("Input path must be a valid file or directory")