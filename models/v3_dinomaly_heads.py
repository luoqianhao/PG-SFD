import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Model-Related Modules
from functools import partial
from models import vit_encoder
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block, Block, LinearAttention2

class DinomalyDetectionModule(nn.Module):
    def __init__(self, embed_dim, num_heads, inp_num, target_layers, 
                 fuse_layer_encoder, fuse_layer_decoder, mask_neighbor_size=0, remove_class_token=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        
        # 异常检测原型
        self.anomaly_prototypes = nn.Parameter(torch.randn(inp_num, embed_dim))
        self.mask_neighbor_size = mask_neighbor_size
        # 聚合模块
        self.aggregation = nn.ModuleList([
            Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            for _ in range(1)
        ])
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)
        ])
        
        # 解码器
        self.decoder = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                          qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                          attn=LinearAttention2)
            for _ in range(8)
        ])
        
        # 特征适配器
        self.feature_adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def gather_loss(self, query, keys):
        distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        distance, cluster_index = torch.min(distribution, dim=2)
        return distance.mean()

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def forward(self, all_encoder_features, encoder_num_register_tokens=0):
        en_list = all_encoder_features

        B = en_list[0].shape[0]

        side = int(math.sqrt(en_list[0].shape[1] - encoder_num_register_tokens - 1))  #28
        
        if self.remove_class_token:
            en_list_processed = [e[:, 1 + encoder_num_register_tokens:, :] for e in en_list]
        else:
            en_list_processed = en_list
        
        # 特征融合
        fused_features = self.fuse_feature(en_list_processed)    #16,784,768
        adapted_features = self.feature_adapter(fused_features)

        # 原型聚合
        agg_prototype = self.anomaly_prototypes
        for blk in self.aggregation:
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), adapted_features)
        
        gather_loss = self.gather_loss(adapted_features, agg_prototype)
        
        # Bottleneck处理
        # bottleneck_features = adapted_features

        for i,blk in enumerate(self.bottleneck):
            bottleneck_features = blk(fused_features)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, bottleneck_features.device)
        else:
            attn_mask = None
        
        # 解码器
        de_list = []
        for blk in self.decoder:
            bottleneck_features = blk(bottleneck_features, attn_mask)
            de_list.append(bottleneck_features)
        de_list = de_list[::-1]
        
        # 特征重构
        en = [self.fuse_feature([en_list_processed[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + encoder_num_register_tokens:, :] for e in en]
            de = [d[:, 1 + encoder_num_register_tokens:, :] for d in de]
        
        # 恢复空间结构
        en = [e.permute(0, 2, 1).reshape([B, -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([B, -1, side, side]).contiguous() for d in de]
        
        return en, de, gather_loss, agg_prototype

class ClassificationModule(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        
        # 分类原型
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))
        
        # 聚合模块
        self.aggregation = nn.ModuleList([
            Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            for _ in range(1)
        ])
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            Mlp(embed_dim, embed_dim * 2, embed_dim, drop=0.2)
        ])

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        # 特征适配器
        self.feature_adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # 分类头
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # 自定义初始化函数
        def init_weights(m):
            if isinstance(m, nn.Linear):
                if m.out_features == self.num_classes:  # 检查是否为最后一层
                    nn.init.normal_(m.weight, std=2.0)  # 使用更大的标准差初始化
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # 应用初始化到分类头的最后一层
        self.classification_head[-1].apply(init_weights)

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def forward(self, en_list, anomaly_prototypes=None, encoder_num_register_tokens=0, remove_class_token=False):
        # 特征处理
        if remove_class_token:
            en_list_processed = [e[:, 1 + encoder_num_register_tokens:, :] for e in en_list]
        else:
            en_list_processed = en_list   #9,10,11层feature
        
        fused_features = self.fuse_feature(en_list_processed)
        adapted_features = self.feature_adapter(fused_features)
        
        # Bottleneck处理
        for blk in self.bottleneck:
            adapted_features = blk(adapted_features)
        
        # 差分增强
        if anomaly_prototypes is not None:
            # 使用 anomaly_module 学到的正常原型作为参考
            p_normal_AD = anomaly_prototypes.detach().mean(1)  # [B, D], 来自无监督学习
            p_normal_AD = p_normal_AD.unsqueeze(1)  # [B, 1, D]，
            f_diff = adapted_features - p_normal_AD   # 偏离“无监督正常”的程度
            gate = self.gate(torch.cat([adapted_features, f_diff], dim=-1))
            adapted_features = adapted_features + gate * f_diff  # 增强特征
            
        # 原型聚合
        B = adapted_features.shape[0]
        class_proto = self.class_prototypes
        for blk in self.aggregation:
            class_proto = blk(class_proto.unsqueeze(0).repeat((B, 1, 1)), adapted_features)
        
        # 全局特征提取
        if remove_class_token:
            global_feature = adapted_features.mean(dim=1)
        else:
            global_feature = adapted_features[:, 0, :]
        
        # 分类输出
        cls_logits = self.classification_head(global_feature)
        
        # 原型相似度
        cls_similarities = F.cosine_similarity(
            global_feature.unsqueeze(1),
            class_proto.unsqueeze(0),
            dim=-1
        )
        
        return cls_logits, cls_similarities, class_proto
    
    
