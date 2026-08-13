import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from timm.models.layers import trunc_normal_

from models.v3_dinomaly_heads import DinomalyDetectionModule, ClassificationModule, ClassificationModule_v2
from models import vit_encoder

class MultiTaskDinomaly(nn.Module):
    def __init__(
            self,
            encoder_name='dinov2reg_vit_small_14',
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[10,11],  # Fine-tune these encoder layers; empty freezes all.
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
        
        # Read embed_dim and num_heads.
        if 'small' in encoder_name:
            embed_dim, num_heads = 384, 6
        elif 'base' in encoder_name:
            embed_dim, num_heads = 768, 12
        elif 'large' in encoder_name:
            embed_dim, num_heads = 1024, 16
            self.target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise "Architecture not in small, base, large."
        
        # Initialize task modules.
        self.anomaly_module = DinomalyDetectionModule(
            embed_dim=embed_dim,
            num_heads=num_heads,
            inp_num=inp_num,
            target_layers=target_layers,
            fuse_layer_encoder=fuse_layer_encoder,
            fuse_layer_decoder=fuse_layer_decoder,
            remove_class_token=remove_class_token
        )
        
        self.classification_module = ClassificationModule_v2(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_classes=num_classes
        )

        self.cls_target_layers = [9, 10, 11]  # Use only deep features.
        
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def extract_encoder_features(self, x):
        """Extract encoder features."""
        x = self.encoder.prepare_tokens(x)
        all_encoder_features = []  # Retain all encoder features; each branch selects its layers.
        finetune_layers = set(self.encoder_require_grad_layer or [])  # Example: {9, 10, 11}.
        for i, blk in enumerate(self.encoder.blocks):
            if len(finetune_layers)==0 or i not in finetune_layers:
                with torch.no_grad():
                    x = blk(x)
            else:
                x = blk(x)        
            all_encoder_features.append(x)
        
        return all_encoder_features

    def forward(self, x, labels=None):
        # Extract encoder features.
        all_encoder_features = self.extract_encoder_features(x)
        
        # Anomaly detection branch.
        anomaly_features = [all_encoder_features[i] for i in self.target_layers]
        en, de, gather_loss, anomaly_prototypes = self.anomaly_module(
            anomaly_features, self.encoder.num_register_tokens
        )
        
        # Classification branch.
        cls_features = [all_encoder_features[i] for i in self.cls_target_layers]

        cls_logits, cls_similarities, class_prototypes = self.classification_module(
            cls_features, anomaly_prototypes,  # Pass anomaly prototypes.
            self.encoder.num_register_tokens, self.remove_class_token
        )
        
        return {
            'anomaly_features': (en, de),
            'gather_loss': gather_loss,
            'cls_logits': cls_logits,
            'cls_similarities': cls_similarities,
            'anomaly_prototypes': anomaly_prototypes,
            'class_prototypes': class_prototypes
        }
    

class MultiTaskDinov3maly(nn.Module):
    def __init__(
            self,
            encoder_name='dinov3_vits16',
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],  # Fine-tune these encoder layers; empty freezes all.
            inp_num=6,
            num_classes=4,
    ) -> None:
        super().__init__()
        self.encoder = vit_encoder.load_dinov3(encoder_name)
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.do_norm = True
        
        # Read embed_dim and num_heads.
        if 's' in encoder_name:
            embed_dim, num_heads = 384, 6
        elif 'b' in encoder_name:
            embed_dim, num_heads = 768, 12
        elif 'l' in encoder_name:
            embed_dim, num_heads = 1024, 16
            self.target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise "Architecture not in small, base, large."
        
        # Initialize task modules.
        self.anomaly_module = DinomalyDetectionModule(
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

        self.cls_target_layers = [9, 10, 11]  # Use only deep features.
        
        if hasattr(self.encoder, 'n_storage_tokens'):
            self.encoder.num_register_tokens = int(self.encoder.n_storage_tokens)
        else:
            self.encoder.num_register_tokens = 0

    def extract_encoder_features(self, x):
        """Extract encoder features."""
        layers_to_take = list(range(self.encoder.n_blocks))
        with torch.no_grad():
            x, (H, W) = self.encoder.prepare_tokens_with_masks(x)
            outputs, total_block_len = [], len(self.encoder.blocks)
            blocks_to_take = range(total_block_len - layers_to_take, total_block_len) if isinstance(layers_to_take, int) else layers_to_take
            for i, blk in enumerate(self.encoder.blocks):
                if self.encoder.rope_embed is not None:
                    rope_sincos = self.encoder.rope_embed(H=H, W=W)
                else:
                    rope_sincos = None
                x = blk(x, rope_sincos)
                if i in blocks_to_take:
                    outputs.append(x)
            all_encoder_features = outputs # Retain all encoder features; each branch selects its layers.
        
        return all_encoder_features

    def forward(self, x):
        # Extract encoder features.
        all_encoder_features = self.extract_encoder_features(x)
        
        # Anomaly detection branch.
        anomaly_features = [all_encoder_features[i] for i in self.target_layers]
        en, de, gather_loss, anomaly_prototypes = self.anomaly_module(
            anomaly_features, self.encoder.num_register_tokens
        )
        
        # Classification branch.
        cls_features = [all_encoder_features[i] for i in self.cls_target_layers]

        cls_logits, cls_similarities, class_prototypes = self.classification_module(
            cls_features, anomaly_prototypes,  # Pass anomaly prototypes.
            self.encoder.num_register_tokens, self.remove_class_token
        )
        
        return {
            'anomaly_features': (en, de),
            'gather_loss': gather_loss,
            'cls_logits': cls_logits,
            'cls_similarities': cls_similarities,
            'anomaly_prototypes': anomaly_prototypes,
            'class_prototypes': class_prototypes
        }
    
class MultiTaskMAEmaly(nn.Module):
    def __init__(
            self,
            encoder_name='mae_vit_base',
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],  # Fine-tune these encoder layers; empty freezes all.
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
        self.do_norm = True
        
        # Read embed_dim and num_heads.
        if 'small' in encoder_name:
            embed_dim, num_heads = 384, 6
        elif 'base' in encoder_name:
            embed_dim, num_heads = 768, 12
        elif 'large' in encoder_name:
            embed_dim, num_heads = 1024, 16
            self.target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise "Architecture not in small, base, large."
        
        # Initialize task modules.
        self.anomaly_module = DinomalyDetectionModule(
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

        self.cls_target_layers = [9, 10, 11]  # Use only deep features.
        
        if hasattr(self.encoder, 'n_storage_tokens'):
            self.encoder.num_register_tokens = int(self.encoder.n_storage_tokens)
        else:
            self.encoder.num_register_tokens = 0

    def extract_encoder_features(self, x):
        """Extract encoder features."""
        all_encoder_features = []
        x = self.encoder.forward_encoder_only(x)
        with torch.no_grad():
            for blk in self.encoder.blocks:
                x = blk(x)  
                all_encoder_features.append(x) # Retain all encoder features; each branch selects its layers.
        
        return all_encoder_features

    def forward(self, x):
        # Extract encoder features.
        all_encoder_features = self.extract_encoder_features(x)
        
        # Anomaly detection branch.
        anomaly_features = [all_encoder_features[i] for i in self.target_layers]
        en, de, gather_loss, anomaly_prototypes = self.anomaly_module(
            anomaly_features, self.encoder.num_register_tokens
        )
        
        # Classification branch.
        cls_features = [all_encoder_features[i] for i in self.cls_target_layers]

        cls_logits, cls_similarities, class_prototypes = self.classification_module(
            cls_features, anomaly_prototypes,  # Pass anomaly prototypes.
            self.encoder.num_register_tokens, self.remove_class_token
        )
        
        return {
            'anomaly_features': (en, de),
            'gather_loss': gather_loss,
            'cls_logits': cls_logits,
            'cls_similarities': cls_similarities,
            'anomaly_prototypes': anomaly_prototypes,
            'class_prototypes': class_prototypes
        }

class MultiTaskIBOTmaly(nn.Module):
    def __init__(
            self,
            encoder_name='ibot_vit_small',
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],  # Fine-tune these encoder layers; empty freezes all.
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
        self.do_norm = True
        
        # Read embed_dim and num_heads.
        if 'small' in encoder_name:
            embed_dim, num_heads = 384, 6
        elif 'base' in encoder_name:
            embed_dim, num_heads = 768, 12
        elif 'large' in encoder_name:
            embed_dim, num_heads = 1024, 16
            self.target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise "Architecture not in small, base, large."
        
        # Initialize task modules.
        self.anomaly_module = DinomalyDetectionModule(
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

        self.cls_target_layers = [9, 10, 11]  # Use only deep features.
        
        if hasattr(self.encoder, 'n_storage_tokens'):
            self.encoder.num_register_tokens = int(self.encoder.n_storage_tokens)
        else:
            self.encoder.num_register_tokens = 0

    def extract_encoder_features(self, x):
        """Extract encoder features."""
        all_encoder_features = []
        x = self.encoder.prepare_tokens(x)
        with torch.no_grad():
            for i, blk in enumerate(self.encoder.blocks):
                x = blk(x)
                all_encoder_features.append(x)
        
        return all_encoder_features

    def forward(self, x):
        # Extract encoder features.
        all_encoder_features = self.extract_encoder_features(x)
        
        # Anomaly detection branch.
        anomaly_features = [all_encoder_features[i] for i in self.target_layers]
        en, de, gather_loss, anomaly_prototypes = self.anomaly_module(
            anomaly_features, self.encoder.num_register_tokens
        )
        
        # Classification branch.
        cls_features = [all_encoder_features[i] for i in self.cls_target_layers]

        cls_logits, cls_similarities, class_prototypes = self.classification_module(
            cls_features, anomaly_prototypes,  # Pass anomaly prototypes.
            self.encoder.num_register_tokens, self.remove_class_token
        )
        
        return {
            'anomaly_features': (en, de),
            'gather_loss': gather_loss,
            'cls_logits': cls_logits,
            'cls_similarities': cls_similarities,
            'anomaly_prototypes': anomaly_prototypes,
            'class_prototypes': class_prototypes
        }

class MultiTaskDeitmaly(nn.Module):
    def __init__(
            self,
            encoder_name='deit_vit_small_16',
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],  # Fine-tune these encoder layers; empty freezes all.
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
        self.do_norm = True
        
        # Read embed_dim and num_heads.
        if 'small' in encoder_name:
            embed_dim, num_heads = 384, 6
        elif 'base' in encoder_name:
            embed_dim, num_heads = 768, 12
        elif 'large' in encoder_name:
            embed_dim, num_heads = 1024, 16
            self.target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise "Architecture not in small, base, large."
        
        # Initialize task modules.
        self.anomaly_module = DinomalyDetectionModule(
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

        self.cls_target_layers = [9, 10, 11]  # Use only deep features.
        
        if hasattr(self.encoder, 'n_storage_tokens'):
            self.encoder.num_register_tokens = int(self.encoder.n_storage_tokens)
        else:
            self.encoder.num_register_tokens = 0

    def extract_encoder_features(self, x):
        """Extract encoder features."""
        all_encoder_features = []
        x = self.encoder.prepare_tokens(x)
        with torch.no_grad():
            for i, blk in enumerate(self.encoder.blocks):
                x = blk(x)
                all_encoder_features.append(x)
        
        return all_encoder_features

    def forward(self, x):
        # Extract encoder features.
        all_encoder_features = self.extract_encoder_features(x)
        
        # Anomaly detection branch.
        anomaly_features = [all_encoder_features[i] for i in self.target_layers]
        en, de, gather_loss, anomaly_prototypes = self.anomaly_module(
            anomaly_features, self.encoder.num_register_tokens
        )
        
        # Classification branch.
        cls_features = [all_encoder_features[i] for i in self.cls_target_layers]

        cls_logits, cls_similarities, class_prototypes = self.classification_module(
            cls_features, anomaly_prototypes,  # Pass anomaly prototypes.
            self.encoder.num_register_tokens, self.remove_class_token
        )
        
        return {
            'anomaly_features': (en, de),
            'gather_loss': gather_loss,
            'cls_logits': cls_logits,
            'cls_similarities': cls_similarities,
            'anomaly_prototypes': anomaly_prototypes,
            'class_prototypes': class_prototypes
        }
