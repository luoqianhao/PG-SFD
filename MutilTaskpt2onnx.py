import os 
import argparse
import torch
from models.my_model import MultiTaskINPFormer
from models.v3_dinomaly import MultiTaskDinomaly
from utils_multiTask import evaluation_batch,WarmCosineScheduler, global_cosine_hm_adaptive, setup_seed, get_logger, evaluation_batch_simple

class Wrapper(torch.nn.Module):
    """
    Wrap original model that returns a dict:
    {
        'anomaly_features': (en, de),
        'gather_loss': gather_loss,
        'cls_logits': cls_logits,
        'cls_similarities': cls_similarities,
        'anomaly_prototypes': anomaly_prototypes,
        'class_prototypes': class_prototypes
    }
    into a tuple of tensors for ONNX export.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    @staticmethod
    def _to_tensor(v, ref: torch.Tensor):
        # Ensure every output is a tensor (ONNX needs tensors)
        if torch.is_tensor(v):
            return v
        # fall back: make a tensor on the same device/dtype as ref
        return torch.as_tensor(v, dtype=ref.dtype, device=ref.device)

    def forward(self, x):
        out = self.model(x)
        en_list = out['anomaly_features'][0]
        de_list = out['anomaly_features'][1]
        en_0 = en_list[0]
        en_1 = en_list[1]
        de_0 = de_list[0]
        de_1 = de_list[1]
        # pick a reference tensor to place scalars
        ref = en_0 if torch.is_tensor(en_0) else (x if torch.is_tensor(x) else torch.tensor(0.0))
        gather_loss = self._to_tensor(out['gather_loss'], ref)
        cls_logits = self._to_tensor(out['cls_logits'], ref)
        cls_similarities = self._to_tensor(out['cls_similarities'], ref)
        anomaly_prototypes = self._to_tensor(out['anomaly_prototypes'], ref)
        class_prototypes = self._to_tensor(out['class_prototypes'], ref)

        return (
            en_0,                         # 0: en
            en_1,                         # 1: de
            de_0,
            de_1,
            gather_loss,                # 2: gather_loss (tensor-ized)
            cls_logits,                 # 3: cls_logits
            cls_similarities,           # 4: cls_similarities
            anomaly_prototypes,         # 5: anomaly_prototypes
            class_prototypes            # 6: class_prototypes
        )

# def load_model(args):
#     model = MultiTaskDinomaly(encoder_name=args.encoder, remove_class_token=True, inp_num=args.INP_num, num_classes=args.num_classes)
#     model.load_state_dict(torch.load(args.checkpoint))
#     model = model.to(device)
#     model.eval()
#     return model

def load_model(args):
    model = MultiTaskDinomaly(
        encoder_name=args.encoder,
        remove_class_token=True,
        inp_num=args.INP_num,
        num_classes=args.num_classes
    )
    # Load safely on the CPU first.
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    # Unwrap checkpoints containing a "model" key.
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)  # Use strict=False for debugging.
    model = model.to(device)
    model.eval()
    return model

def export_onnx(model,args):

    dummy_input = torch.randn(1, 3, args.crop_size, args.crop_size, dtype=torch.float32).to(device)  
    torch.onnx.export(
        model, 
        dummy_input,              # For multiple inputs, use (dummy_img, dummy_mask).
        args.save_name,
        input_names=["image"],    # For multiple inputs, use ["image", "mask"].
        output_names = ["en_0","en_1",'de_0','de_1',"gather_loss","cls_logits","cls_similarities","anomaly_prototypes","class_prototypes"],  # For multiple outputs, use ["cls", "mask"].
        dynamic_axes = {
        "image": {0: "batch"},
        "en_0": {0: "batch"},
        "en_1": {0: "batch"},
        "de_0": {0: "batch"},
        "de_1": {0: "batch"},

    },
        opset_version=16,         # Use a recent ONNX opset.
        do_constant_folding=True,
    )
    print("✅ ONNX 模型已导出: model.onnx")

if __name__ == '__main__':
    setup_seed(1)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "7"
    parser = argparse.ArgumentParser(description='pt2onnx')

    # model info
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_small_14') # 'dinov2reg_vit_small_14' or 'dinov2reg_vit_base_14' or 'dinov2reg_vit_large_14'
    parser.add_argument('--checkpoint', type=str, default="/home/cisdi/luoqianhao/PSG-FSD/saved_results/mutiTaskv3Dinomaly_gangguan-20251108_lqh/model_epoch_200_gangguan_2d_multiTask_1208.pth")
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--crop_size', type=int, default=448)
    parser.add_argument('--INP_num', type=int, default=6)
    parser.add_argument('--num_classes', type=int, default=3) # only for multi-class 
    parser.add_argument('--save_name', type=str, default='/home/cisdi/luoqianhao/PSG-FSD/saved_results/mutiTaskv3Dinomaly_gangguan-20251108_lqh/mutitaskv3dinomaly_1208.onnx')


    args = parser.parse_args()
    device = 'cuda:7' if torch.cuda.is_available() else 'cpu'

    model = load_model(args)
    model = Wrapper(model).to(device).eval()
    export_onnx(model, args)
