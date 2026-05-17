import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from timm.models.layers import trunc_normal_

from models.heads import AnomalyDetectionModule,AnomalyDetectionModule_v2, ClassificationModule, ClassificationModule_v2, ClassificationModule_v3,AnomalyDetectionModule_v3
from models import vit_encoder

class MultiTaskINPFormer(nn.Module):
    def __init__(
            self,
            encoder_name='dinov2reg_vit_small_14',
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],  # encoder中需要微调的层，输入为空的话表示全部冻结
            inp_num=6,
            num_classes=4,
    ) -> None:
        super().__init__()
        self.encoder = vit_encoder.load(encoder_name)
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        
        # 获取embed_dim和num_heads
        if 'small' in encoder_name:
            embed_dim, num_heads = 384, 6
        elif 'base' in encoder_name:
            embed_dim, num_heads = 768, 12
        elif 'large' in encoder_name:
            embed_dim, num_heads = 1024, 16
            target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise "Architecture not in small, base, large."
        
        # 初始化任务模块
        self.anomaly_module = AnomalyDetectionModule(
            embed_dim=embed_dim,
            num_heads=num_heads,
            inp_num=inp_num,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            remove_class_token=remove_class_token
        )
        
        self.classification_module = ClassificationModule(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_classes=num_classes
        )

        self.cls_target_layers = [9, 10, 11]  # 只要深层次的特征即可
        
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def extract_encoder_features(self, x):
        """提取编码器特征"""
        x = self.encoder.prepare_tokens(x)
        all_encoder_features = []  # 存储所有encoder层的特征,筛选下放到子任务中进行

        for i, blk in enumerate(self.encoder.blocks):
            with torch.no_grad():
                x = blk(x)
            all_encoder_features.append(x)
        
        return all_encoder_features

    def forward(self, x):
        # 提取编码器特征
        all_encoder_features = self.extract_encoder_features(x)
        
        # 异常检测任务
        anomaly_features = [all_encoder_features[i] for i in self.target_layers]
        en, de, gather_loss, anomaly_prototypes = self.anomaly_module(
            anomaly_features, self.encoder.num_register_tokens
        )
        
        # 分类任务
        cls_features = [all_encoder_features[i] for i in self.cls_target_layers]
        # cls_logits, cls_similarities, class_prototypes = self.classification_module(
        #     cls_features, self.encoder.num_register_tokens, self.remove_class_token
        # )
        cls_logits, cls_similarities, class_prototypes = self.classification_module(
            cls_features, anomaly_prototypes,  # 传入异常原型
            self.encoder.num_register_tokens, self.remove_class_token
        )
        # cls_logits = self.classification_module(
        #     cls_features, anomaly_prototypes,  # 传入异常原型
        #     self.encoder.num_register_tokens, self.remove_class_token
        # )
        
        return {
            'anomaly_features': (en, de),
            'gather_loss': gather_loss,
            'cls_logits': cls_logits,
            'cls_similarities': cls_similarities,
            'anomaly_prototypes': anomaly_prototypes,
            'class_prototypes': class_prototypes
        }
    
class MultiTaskINPFormer_v2(nn.Module):
    def __init__(
            self,
            encoder_name='dinov2reg_vit_small_14',
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],  # encoder中需要微调的层，输入为空的话表示全部冻结
            inp_num=6,
            num_classes=4,
    ) -> None:
        super().__init__()
        self.encoder = vit_encoder.load(encoder_name)
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        
        # 获取embed_dim和num_heads
        if 'small' in encoder_name:
            embed_dim, num_heads = 384, 6
        elif 'base' in encoder_name:
            embed_dim, num_heads = 768, 12
        elif 'large' in encoder_name:
            embed_dim, num_heads = 1024, 16
            target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise "Architecture not in small, base, large."
        
        # 初始化任务模块
        self.anomaly_module = AnomalyDetectionModule_v2(
            embed_dim=embed_dim,
            num_heads=num_heads,
            inp_num=inp_num,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            remove_class_token=remove_class_token
        )
        
        self.classification_module = ClassificationModule(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_classes=num_classes
        )

        self.cls_target_layers = [9, 10, 11]  # 只要深层次的特征即可
        
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def extract_encoder_features(self, x):
        """提取编码器特征"""
        x = self.encoder.prepare_tokens(x)
        all_encoder_features = []  # 存储所有encoder层的特征,筛选下放到子任务中进行

        for i, blk in enumerate(self.encoder.blocks):
            with torch.no_grad():
                x = blk(x)
            all_encoder_features.append(x)
        
        return all_encoder_features

    def forward(self, x):
        # 提取编码器特征
        all_encoder_features = self.extract_encoder_features(x)
        
        # 异常检测任务
        anomaly_features = [all_encoder_features[i] for i in self.target_layers]
        en, de, gather_loss, anomaly_prototypes = self.anomaly_module(
            anomaly_features, self.encoder.num_register_tokens
        )
        
        # 分类任务
        cls_features = [all_encoder_features[i] for i in self.cls_target_layers]
        # cls_logits, cls_similarities, class_prototypes = self.classification_module(
        #     cls_features, self.encoder.num_register_tokens, self.remove_class_token
        # )
        cls_logits, cls_similarities, class_prototypes = self.classification_module(
            cls_features, anomaly_prototypes,  # 传入异常原型
            self.encoder.num_register_tokens, self.remove_class_token
        )
        # cls_logits = self.classification_module(
        #     cls_features, anomaly_prototypes,  # 传入异常原型
        #     self.encoder.num_register_tokens, self.remove_class_token
        # )
        
        return {
            'anomaly_features': (en, de),
            'gather_loss': gather_loss,
            'cls_logits': cls_logits,
            'cls_similarities': cls_similarities,
            'anomaly_prototypes': anomaly_prototypes,
            'class_prototypes': class_prototypes
        }
    
class MultiTaskINPFormer_v3(nn.Module):
    def __init__(
            self,
            encoder_name='dinov2reg_vit_small_14',
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],  # encoder中需要微调的层，输入为空的话表示全部冻结
            inp_num=6,
            num_classes=4,
    ) -> None:
        super().__init__()
        self.encoder = vit_encoder.load(encoder_name)
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        
        # 获取embed_dim和num_heads
        if 'small' in encoder_name:
            embed_dim, num_heads = 384, 6
        elif 'base' in encoder_name:
            embed_dim, num_heads = 768, 12
        elif 'large' in encoder_name:
            embed_dim, num_heads = 1024, 16
            target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise "Architecture not in small, base, large."
        
        # 初始化任务模块
        self.anomaly_module = AnomalyDetectionModule_v3(
            embed_dim=embed_dim,
            num_heads=num_heads,
            inp_num=inp_num,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            remove_class_token=remove_class_token
        )
        
        self.classification_module = ClassificationModule(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_classes=num_classes
        )

        self.cls_target_layers = [9, 10, 11]  # 只要深层次的特征即可
        
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def extract_encoder_features(self, x):
        """提取编码器特征"""
        x = self.encoder.prepare_tokens(x)
        all_encoder_features = []  # 存储所有encoder层的特征,筛选下放到子任务中进行

        for i, blk in enumerate(self.encoder.blocks):
            with torch.no_grad():
                x = blk(x)
            all_encoder_features.append(x)
        
        return all_encoder_features

    def forward(self, x):
        # 提取编码器特征
        all_encoder_features = self.extract_encoder_features(x)
        
        # 异常检测任务
        anomaly_features = [all_encoder_features[i] for i in self.target_layers]
        en, de, gather_loss, anomaly_prototypes = self.anomaly_module(
            anomaly_features, self.encoder.num_register_tokens
        )
        
        # 分类任务
        cls_features = [all_encoder_features[i] for i in self.cls_target_layers]
        # cls_logits, cls_similarities, class_prototypes = self.classification_module(
        #     cls_features, self.encoder.num_register_tokens, self.remove_class_token
        # )
        cls_logits, cls_similarities, class_prototypes = self.classification_module(
            cls_features, anomaly_prototypes,  # 传入异常原型
            self.encoder.num_register_tokens, self.remove_class_token
        )
        # cls_logits = self.classification_module(
        #     cls_features, anomaly_prototypes,  # 传入异常原型
        #     self.encoder.num_register_tokens, self.remove_class_token
        # )
        
        return {
            'anomaly_features': (en, de),
            'gather_loss': gather_loss,
            'cls_logits': cls_logits,
            'cls_similarities': cls_similarities,
            'anomaly_prototypes': anomaly_prototypes,
            'class_prototypes': class_prototypes
        }
    
class MultiTaskINPFormer_v4(nn.Module):
    def __init__(
            self,
            encoder_name='dinov2reg_vit_small_14',
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],  # encoder中需要微调的层，输入为空的话表示全部冻结
            inp_num=6,
            num_classes=4,
    ) -> None:
        super().__init__()
        self.encoder = vit_encoder.load(encoder_name)
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        
        # 获取embed_dim和num_heads
        if 'small' in encoder_name:
            embed_dim, num_heads = 384, 6
        elif 'base' in encoder_name:
            embed_dim, num_heads = 768, 12
        elif 'large' in encoder_name:
            embed_dim, num_heads = 1024, 16
            target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise "Architecture not in small, base, large."
        
        # 初始化任务模块
        self.anomaly_module = AnomalyDetectionModule_v4(
            embed_dim=embed_dim,
            num_heads=num_heads,
            inp_num=inp_num,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            remove_class_token=remove_class_token
        )
        
        self.classification_module = ClassificationModule(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_classes=num_classes
        )

        self.cls_target_layers = [9, 10, 11]  # 只要深层次的特征即可
        
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def extract_encoder_features(self, x):
        """提取编码器特征"""
        x = self.encoder.prepare_tokens(x)
        all_encoder_features = []  # 存储所有encoder层的特征,筛选下放到子任务中进行

        for i, blk in enumerate(self.encoder.blocks):
            with torch.no_grad():
                x = blk(x)
            all_encoder_features.append(x)
        
        return all_encoder_features

    def forward(self, x):
        # 提取编码器特征
        all_encoder_features = self.extract_encoder_features(x)
        
        # 异常检测任务
        anomaly_features = [all_encoder_features[i] for i in self.target_layers]
        en, de, gather_loss, anomaly_prototypes = self.anomaly_module(
            anomaly_features, self.encoder.num_register_tokens
        )
        
        # 分类任务
        cls_features = [all_encoder_features[i] for i in self.cls_target_layers]
        # cls_logits, cls_similarities, class_prototypes = self.classification_module(
        #     cls_features, self.encoder.num_register_tokens, self.remove_class_token
        # )
        cls_logits, cls_similarities, class_prototypes = self.classification_module(
            cls_features, anomaly_prototypes,  # 传入异常原型
            self.encoder.num_register_tokens, self.remove_class_token
        )
        # cls_logits = self.classification_module(
        #     cls_features, anomaly_prototypes,  # 传入异常原型
        #     self.encoder.num_register_tokens, self.remove_class_token
        # )
        
        return {
            'anomaly_features': (en, de),
            'gather_loss': gather_loss,
            'cls_logits': cls_logits,
            'cls_similarities': cls_similarities,
            'anomaly_prototypes': anomaly_prototypes,
            'class_prototypes': class_prototypes
        }