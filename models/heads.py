import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Model-Related Modules
from functools import partial
from models import vit_encoder
from models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block

# Reproduce InPFormer's anomaly detector.
class AnomalyDetectionModule(nn.Module):
    def __init__(self, embed_dim, num_heads, inp_num, target_layers, 
                 fuse_layer_encoder, fuse_layer_decoder, remove_class_token=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        
        # Anomaly prototypes.
        self.anomaly_prototypes = nn.Parameter(torch.randn(inp_num, embed_dim))
        
        # Aggregation module.
        self.aggregation = nn.ModuleList([
            Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            for _ in range(1)
        ])
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)
        ])
        
        # Decoder.
        self.decoder = nn.ModuleList([
            Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                          qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            for _ in range(8)
        ])
        
        # Feature adapter.
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

        side = int(math.sqrt(en_list[0].shape[1] - encoder_num_register_tokens - 1))
        
        if self.remove_class_token:
            en_list_processed = [e[:, 1 + encoder_num_register_tokens:, :] for e in en_list]
        else:
            en_list_processed = en_list
        
        
        # Fuse features.
        fused_features = self.fuse_feature(en_list_processed)
        adapted_features = self.feature_adapter(fused_features)
        
        # Aggregate prototypes.
        agg_prototype = self.anomaly_prototypes
        for blk in self.aggregation:
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), adapted_features)
        
        gather_loss = self.gather_loss(adapted_features, agg_prototype)
        
        # Bottleneck.
        bottleneck_features = adapted_features
        for blk in self.bottleneck:
            bottleneck_features = blk(bottleneck_features)
        
        # Decoder.
        de_list = []
        for blk in self.decoder:
            bottleneck_features = blk(bottleneck_features, agg_prototype)
            de_list.append(bottleneck_features)
        de_list = de_list[::-1]
        
        # Reconstruct features.
        en = [self.fuse_feature([en_list_processed[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + encoder_num_register_tokens:, :] for e in en]
            de = [d[:, 1 + encoder_num_register_tokens:, :] for d in de]
        
        # Restore spatial layout.
        en = [e.permute(0, 2, 1).reshape([B, -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([B, -1, side, side]).contiguous() for d in de]
        
        return en, de, gather_loss, agg_prototype

class ClassificationModule(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        
        # Class prototypes.
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))

        # Difference enhancement module (TODO).
        # self.differential_enhance = DifferentialEnhancement_v1_Block(embed_dim)
        
        # Aggregation module.
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
        
        # Feature adapter.
        self.feature_adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Classification head.
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Custom initialization.
        def init_weights(m):
            if isinstance(m, nn.Linear):
                if m.out_features == self.num_classes:  # Check for the final layer.
                    nn.init.normal_(m.weight, std=2.0)  # Initialize with a larger standard deviation.
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # Initialize the final classification layer.
        self.classification_head[-1].apply(init_weights)

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def forward(self, en_list, anomaly_prototypes=None, encoder_num_register_tokens=0, remove_class_token=False):
        # Process features.
        if remove_class_token:
            en_list_processed = [e[:, 1 + encoder_num_register_tokens:, :] for e in en_list]
        else:
            en_list_processed = en_list   # Features from layers 9, 10, and 11.
        
        fused_features = self.fuse_feature(en_list_processed)
        adapted_features = self.feature_adapter(fused_features)
        
        # Bottleneck.
        for blk in self.bottleneck:
            adapted_features = blk(adapted_features)
        
        # Difference enhancement.
        if anomaly_prototypes is not None:
            # Use the normal prototype learned by anomaly_module.
            p_normal_AD = anomaly_prototypes.mean(1)  # [B, D], learned without supervision.
            p_normal_AD = p_normal_AD.unsqueeze(1)  # [B, 1, D]，
            f_diff = adapted_features - p_normal_AD   # Distance from the learned normal pattern.
            gate = self.gate(torch.cat([adapted_features, f_diff], dim=-1))
            adapted_features = adapted_features + gate * f_diff  # Enhance features.
            
        # Aggregate prototypes.
        B = adapted_features.shape[0]
        class_proto = self.class_prototypes
        for blk in self.aggregation:
            class_proto = blk(class_proto.unsqueeze(0).repeat((B, 1, 1)), adapted_features)
        
        # Extract global features.
        if remove_class_token:
            global_feature = adapted_features.mean(dim=1)
        else:
            global_feature = adapted_features[:, 0, :]
        
        # Classification output.
        cls_logits = self.classification_head(global_feature)
        
        # Prototype similarity.
        cls_similarities = F.cosine_similarity(
            global_feature.unsqueeze(1),
            class_proto.unsqueeze(0),
            dim=-1
        )
        
        return cls_logits, cls_similarities, class_proto
    

class ClassificationModule_v2(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes,temperature_init=20.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        
        # Class prototypes.
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))

        # Difference enhancement module (TODO).
        # self.differential_enhance = DifferentialEnhancement_v1_Block(embed_dim)
        
        # Aggregation module.
        self.aggregation = nn.ModuleList([
            Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        ])
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            Mlp(embed_dim, embed_dim * 2, embed_dim, drop=0.0)
        )

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        # Feature adapter.
        self.feature_adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Classification head.
        # self.classification_head = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, embed_dim // 2),
        #     nn.GELU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(embed_dim // 2, num_classes)
        # )
        self.scale = nn.Parameter(torch.tensor(temperature_init))  # Analogous to ArcFace scale s.
        self.norm = nn.LayerNorm(embed_dim)
        
        # # Custom initialization.
        # def init_weights(m):
        #     if isinstance(m, nn.Linear):
        #         if m.out_features == self.num_classes:  # Check for the final layer.
        #             nn.init.normal_(m.weight, std=2.0)  # Use a larger standard deviation.
        #             if m.bias is not None:
        #                 nn.init.constant_(m.bias, 0)

        # # Initialize the final classification layer.
        # self.classification_head[-1].apply(init_weights)

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def forward(self, en_list, anomaly_prototypes=None, encoder_num_register_tokens=0, remove_class_token=False):
        # Process features.
        if remove_class_token:
            en_list_processed = [e[:, 1 + encoder_num_register_tokens:, :] for e in en_list]
        else:
            en_list_processed = en_list   # Features from layers 9, 10, and 11.
        
        fused_features = self.fuse_feature(en_list_processed)
        adapted_features = self.feature_adapter(fused_features)
        
        # Bottleneck.
        adapted_features = self.bottleneck(adapted_features)
        
        # Difference enhancement.
        if anomaly_prototypes is not None:
            # Use the normal prototype learned by anomaly_module.
            p_normal_AD = anomaly_prototypes.mean(1)  # [B, D], learned without supervision.
            p_normal_AD = p_normal_AD.unsqueeze(1)  # [B, 1, D]，
            f_diff = adapted_features - p_normal_AD   # Distance from the learned normal pattern.
            gate = self.gate(torch.cat([adapted_features, f_diff], dim=-1))
            adapted_features = adapted_features + gate * f_diff  # Enhance features.
            
        # Aggregate prototypes.
        B = adapted_features.shape[0]
        class_proto = self.class_prototypes
        for blk in self.aggregation:
            class_proto = blk(class_proto.unsqueeze(0).expand((B, -1, -1)), adapted_features)
        
        # Extract global features.
        if remove_class_token:
            global_feature = adapted_features.mean(dim=1)
        else:
            global_feature = adapted_features[:, 0, :]
        
        # Classification output.
        # cls_logits = self.classification_head(global_feature)
        feat = F.normalize(global_feature, dim=-1)                  # [B, D]
        proto = F.normalize(class_proto, dim=-1)                 # [B, C, D]
        cls_logits = self.scale * torch.einsum('bd,bcd->bc', feat, proto)
        cls_similarities = cls_logits / self.scale 
        # Prototype similarity.
        # cls_similarities = F.cosine_similarity(
        #     global_feature.unsqueeze(1),
        #     class_proto.unsqueeze(0),
        #     dim=-1
        # )
        class_proto = class_proto if class_proto.dim() == 3 else self.class_prototypes
        
        return cls_logits, cls_similarities, class_proto
    

class ClassificationModule_v3(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes,temperature_init=20.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        
        # Class prototypes.
        self.class_prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))

        # Difference enhancement module (TODO).
        # self.differential_enhance = DifferentialEnhancement_v1_Block(embed_dim)
        
        # Aggregation module.
        self.aggregation = nn.ModuleList([
            Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
        ])
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            Mlp(embed_dim, embed_dim * 2, embed_dim, drop=0.0)
        )

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        # Feature adapter.
        self.feature_adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Classification head.
        self.classification_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes)
        )
      
        
        # Custom initialization.
        def init_weights(m):
            if isinstance(m, nn.Linear):
                if m.out_features == self.num_classes:  # Check for the final layer.
                    nn.init.normal_(m.weight, std=2.0)  # Initialize with a larger standard deviation.
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        # Initialize the final classification layer.
        self.classification_head[-1].apply(init_weights)

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def forward(self, en_list, anomaly_prototypes=None, encoder_num_register_tokens=0, remove_class_token=False):
        # Process features.
        if remove_class_token:
            en_list_processed = [e[:, 1 + encoder_num_register_tokens:, :] for e in en_list]
        else:
            en_list_processed = en_list   # Features from layers 9, 10, and 11.
        
        fused_features = self.fuse_feature(en_list_processed)
        adapted_features = self.feature_adapter(fused_features)
        
        # Extract global features.
        if remove_class_token:
            global_feature = adapted_features.mean(dim=1)
        else:
            global_feature = adapted_features[:, 0, :]
        
        # Classification output.
        cls_logits = self.classification_head(global_feature)
       
        # # Prototype similarity.
        # cls_similarities = F.cosine_similarity(
        #     global_feature.unsqueeze(1),
        #     class_proto.unsqueeze(0),
        #     dim=-1
        # )
        # class_proto = class_proto if class_proto.dim() == 3 else self.class_prototypes
        
        return cls_logits

# V2 adds normalized features and prototypes.
class AnomalyDetectionModule_v2(nn.Module):
    def __init__(self, embed_dim, num_heads, inp_num, target_layers, 
                 fuse_layer_encoder, fuse_layer_decoder, remove_class_token=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        
        # Anomaly prototypes.
        self.anomaly_prototypes = nn.Parameter(torch.randn(inp_num, embed_dim))
                
        # Aggregation module.
        self.aggregation = nn.ModuleList([
            Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            for _ in range(1)
        ])
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)
        ])
        
        # Decoder.
        self.decoder = nn.ModuleList([
            Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                          qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            for _ in range(8)
        ])
        
        # Feature adapter.
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

        side = int(math.sqrt(en_list[0].shape[1] - encoder_num_register_tokens - 1))
        
        if self.remove_class_token:
            en_list_processed = [e[:, 1 + encoder_num_register_tokens:, :] for e in en_list]
        else:
            en_list_processed = en_list
        
        
        # Fuse features.
        fused_features = self.fuse_feature(en_list_processed)
        adapted_features = self.feature_adapter(fused_features)
        
        # Aggregate prototypes.
        agg_prototype = self.anomaly_prototypes
        # Add normalized features and prototypes.
        agg_prototype = F.normalize(agg_prototype, dim=-1)
        adapted_features = F.normalize(adapted_features, dim=-1)
        
        for blk in self.aggregation:
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), adapted_features)
        
        gather_loss = self.gather_loss(fused_features, agg_prototype)
        
        # Bottleneck.
        bottleneck_features = adapted_features
        for blk in self.bottleneck:
            bottleneck_features = blk(bottleneck_features)
        
        # Decoder.
        de_list = []
        for blk in self.decoder:
            bottleneck_features = blk(bottleneck_features, agg_prototype)
            de_list.append(bottleneck_features)
        de_list = de_list[::-1]
        
        # Reconstruct features.
        en = [self.fuse_feature([en_list_processed[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + encoder_num_register_tokens:, :] for e in en]
            de = [d[:, 1 + encoder_num_register_tokens:, :] for d in de]
        
        # Restore spatial layout.
        en = [e.permute(0, 2, 1).reshape([B, -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([B, -1, side, side]).contiguous() for d in de]
        
        return en, de, gather_loss, agg_prototype

# V3 adds reconstruction loss and normalization.
class AnomalyDetectionModule_v3(nn.Module):
    def __init__(self, embed_dim, num_heads, inp_num, target_layers, 
                 fuse_layer_encoder, fuse_layer_decoder, remove_class_token=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        
        # Anomaly prototypes.
        self.anomaly_prototypes = nn.Parameter(torch.randn(inp_num, embed_dim))
        
        # Aggregation module.
        self.aggregation = nn.ModuleList([
            Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            for _ in range(1)
        ])
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.)
        ])
        
        # Decoder.
        self.decoder = nn.ModuleList([
            Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                          qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            for _ in range(8)
        ])
        
        # Feature adapter.
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

        side = int(math.sqrt(en_list[0].shape[1] - encoder_num_register_tokens - 1))
        
        if self.remove_class_token:
            en_list_processed = [e[:, 1 + encoder_num_register_tokens:, :] for e in en_list]
        else:
            en_list_processed = en_list
        
        
        # Fuse features.
        fused_features = self.fuse_feature(en_list_processed)
        adapted_features = self.feature_adapter(fused_features)
        
        # Aggregate prototypes.
        agg_prototype = self.anomaly_prototypes
        # Normalize.
        agg_prototype = F.normalize(agg_prototype, dim=-1)
        adapted_features = F.normalize(adapted_features, dim=-1)

        for blk in self.aggregation:
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), adapted_features)
        
        gather_loss = self.gather_loss(adapted_features, agg_prototype)
        
        # Bottleneck.
        bottleneck_features = adapted_features
        for blk in self.bottleneck:
            bottleneck_features = blk(bottleneck_features)
        
        # Decoder.
        de_list = []
        for blk in self.decoder:
            bottleneck_features = blk(bottleneck_features, agg_prototype)
            de_list.append(bottleneck_features)
        de_list = de_list[::-1]
        
        # Reconstruct features.
        en = [self.fuse_feature([en_list_processed[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + encoder_num_register_tokens:, :] for e in en]
            de = [d[:, 1 + encoder_num_register_tokens:, :] for d in de]
        
        # Restore spatial layout.
        en = [e.permute(0, 2, 1).reshape([B, -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([B, -1, side, side]).contiguous() for d in de]

        recon_loss = F.l1_loss(torch.stack(de, dim=1), torch.stack(en, dim=1))  # Reconstruction loss.
        gather_loss = gather_loss + 0.5 * recon_loss
        
        return en, de, gather_loss, agg_prototype
