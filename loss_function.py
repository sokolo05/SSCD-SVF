#!/usr/bin/env python3
"""
Confidence-Driven Collaborative Denoising (CDCD) Loss Function Module - Fixed Version
Only contains loss function specific parts, common modules removed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from typing import Optional, Tuple, Dict, List, Any
import pandas as pd
import numpy as np
from pathlib import Path

# Import from common modules
from modules import (
    build_backbone_with_layered_finetuning,
    AdaptiveFeatureProjection,
    CrossAttentionFusion,
    AdaptiveConfidenceMapEstimation,
    ConfidenceWeightedCrossAttention,
    AdaptiveCDCDModule
)


# =========================================================
# 1. Adaptive Dual-Modal Model (Fixed Device Issues)
# =========================================================

class AdaptiveCDCDDualModalModel(nn.Module):
    """Adaptive GPK dimension CDCD dual-modal model"""
    def __init__(self, 
                 bases_model_name: str,
                 cigar_model_name: str,
                 num_classes: int = 3,
                 freeze_layers: int = 5,
                 trainable_layers: int = 3,
                 projection_dim: int = 512,
                 gpk_dim: int = 64,
                 use_cdcd: bool = True,
                 use_mac: bool = True,
                 init_alpha: float = 0.1,
                 init_beta: float = 0.1,
                 temperature: float = 0.07,
                 gamma: float = 1.0,
                 rank: int = 0):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_cdcd = use_cdcd
        self.use_mac = use_mac
        self.gpk_dim = gpk_dim
        self.rank = rank
        self.actual_gpk_dim = None
        
        # Build backbones
        self.bases_backbone = build_backbone_with_layered_finetuning(
            bases_model_name, num_classes, freeze_layers, trainable_layers
        )
        self.cigar_backbone = build_backbone_with_layered_finetuning(
            cigar_model_name, num_classes, freeze_layers, trainable_layers
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy = torch.rand(1, 3, 224, 224)
            feature_dim_bases = self.bases_backbone.get_feature_dim(dummy)
            feature_dim_cigar = self.cigar_backbone.get_feature_dim(dummy)
            
            if rank == 0:
                print(f"[INFO] Bases feature dim: {feature_dim_bases}")
                print(f"[INFO] Cigar feature dim: {feature_dim_cigar}")
        
        # Adaptive feature projection
        self.feature_projection = AdaptiveFeatureProjection(
            feature_dim_bases, feature_dim_cigar, projection_dim
        )
        
        # Create placeholders first
        self.cdcd_module = None
        self.classifier = None
        self.mac_loss = None
        
        # Save parameters
        self.init_params = {
            'projection_dim': projection_dim,
            'use_cdcd': use_cdcd,
            'use_mac': use_mac,
            'num_classes': num_classes,
            'init_alpha': init_alpha,
            'init_beta': init_beta,
            'temperature': temperature,
            'gamma': gamma
        }
    
    def initialize_with_actual_gpk_dim(self, actual_gpk_dim: int):
        """Initialize modules with actual GPK dimension (fixed device issues)"""
        self.actual_gpk_dim = actual_gpk_dim
        
        if self.rank == 0:
            print(f"[INFO] Initializing model with actual GPK dimension: {actual_gpk_dim}")
        
        # Get current device
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
        
        if self.use_cdcd:
            self.cdcd_module = AdaptiveCDCDModule(
                feature_dim=self.init_params['projection_dim'],
                gpk_dim=actual_gpk_dim,
                num_heads=8,
                hidden_dim=128
            ).to(device)  # Key fix: Ensure module is on correct device
            
            self.classifier = nn.Sequential(
                nn.Linear(self.init_params['projection_dim'], 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, self.num_classes)
            ).to(device)  # Key fix: Ensure module is on correct device
        else:
            self.cross_attn = CrossAttentionFusion(self.init_params['projection_dim']).to(device)
            
            self.classifier = nn.Sequential(
                nn.Linear(self.init_params['projection_dim'] * 2, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, self.num_classes)
            ).to(device)
        
        if self.use_mac:
            self.mac_loss = MACLoss(
                num_classes=self.num_classes,
                init_alpha=self.init_params['init_alpha'],
                init_beta=self.init_params['init_beta'],
                temperature=self.init_params['temperature'],
                gamma=self.init_params['gamma'],
                use_pmcl=True,
                use_lcon=True,
                weight_constraint='softplus',
                weight_reg_lambda=0.01
            ).to(device)
            
            self.aux_classifier_str = nn.Linear(self.init_params['projection_dim'], self.num_classes).to(device)
            self.aux_classifier_seq = nn.Linear(self.init_params['projection_dim'], self.num_classes).to(device)
            
            self.feature_rectifier_str = nn.Sequential(
                nn.Linear(self.init_params['projection_dim'], self.init_params['projection_dim']),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.LayerNorm(self.init_params['projection_dim'])
            ).to(device)
            
            self.feature_rectifier_seq = nn.Sequential(
                nn.Linear(self.init_params['projection_dim'], self.init_params['projection_dim']),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.LayerNorm(self.init_params['projection_dim'])
            ).to(device)
    
    def forward(self, 
                x1: torch.Tensor,
                x2: torch.Tensor,
                gpk_vector: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        
        if self.use_cdcd and self.cdcd_module is None:
            raise RuntimeError("Model not initialized! Please call initialize_with_actual_gpk_dim() first.")
        
        # Extract features
        feat1 = self.bases_backbone(x1)
        feat2 = self.cigar_backbone(x2)
        
        # Feature projection
        feat1_proj, feat2_proj = self.feature_projection(feat1, feat2)
        
        outputs = {
            'feat1_proj': feat1_proj,
            'feat2_proj': feat2_proj
        }
        
        if self.use_cdcd and gpk_vector is not None:
            # Add sequence dimension
            feat1_seq = feat1_proj.unsqueeze(1)
            feat2_seq = feat2_proj.unsqueeze(1)
            
            H_fusion, confidence_str, confidence_seq = self.cdcd_module(
                H_str=feat1_seq,
                H_seq=feat2_seq,
                gpk_vector=gpk_vector
            )
            
            H_fusion_pooled = H_fusion.mean(dim=1)
            primary_logits = self.classifier(H_fusion_pooled)
            
            outputs.update({
                'primary_logits': primary_logits,
                'fusion_features': H_fusion_pooled,
                'confidence_str': confidence_str,
                'confidence_seq': confidence_seq
            })
        else:
            fused_feat = self.cross_attn(feat1_proj, feat2_proj)
            primary_logits = self.classifier(fused_feat)
            
            outputs.update({
                'primary_logits': primary_logits,
                'fusion_features': fused_feat
            })
        
        if self.training and self.use_mac:
            rectified_str = self.feature_rectifier_str(feat1_proj)
            rectified_seq = self.feature_rectifier_seq(feat2_proj)
            
            outputs.update({
                'rectified_str': rectified_str,
                'rectified_seq': rectified_seq
            })
        
        if not return_features:
            return outputs['primary_logits']
        
        return outputs
    
    def compute_mac_loss(self,
                         outputs: Dict[str, torch.Tensor],
                         labels: torch.Tensor,
                         gpk_vectors: Optional[torch.Tensor] = None,
                         step: int = 0,
                         return_components: bool = False):
        
        if not self.use_mac:
            task_loss = F.cross_entropy(outputs['primary_logits'], labels)
            if return_components:
                return {'task': task_loss, 'total': task_loss}
            return task_loss
        
        loss_dict = self.mac_loss(
            fusion_features=outputs['fusion_features'],
            primary_logits=outputs['primary_logits'],
            labels=labels,
            GPK_vectors=gpk_vectors,
            str_features=outputs.get('rectified_str'),
            seq_features=outputs.get('rectified_seq'),
            return_components=return_components
        )
        
        return loss_dict


# =========================================================
# 2. Loss Function Components
# =========================================================

class PriorModulatedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, gamma: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.gamma = gamma
    
    def compute_kernel_matrix(self, GPK_vectors: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(GPK_vectors, GPK_vectors, p=2)
        kernel_matrix = torch.exp(-self.gamma * distances)
        kernel_matrix = kernel_matrix.fill_diagonal_(0)
        return kernel_matrix
    
    def forward(self, features: torch.Tensor, GPK_vectors: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        N = features.size(0)
        features_norm = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features_norm, features_norm.T) / self.temperature
        kernel_matrix = self.compute_kernel_matrix(GPK_vectors)
        
        if labels is not None:
            label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
            label_mask.fill_diagonal_(0)
            positive_mask = label_mask * kernel_matrix
        else:
            positive_mask = kernel_matrix
        
        numerator = torch.sum(positive_mask * torch.exp(similarity_matrix), dim=1)
        denominator = torch.sum(kernel_matrix * torch.exp(similarity_matrix), dim=1)
        denominator = torch.clamp(denominator, min=1e-8)
        
        loss = -torch.log(numerator / denominator).mean()
        return loss


class LogicConsistencyConstraint(nn.Module):
    def __init__(self):
        super().__init__()
    
    def js_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p = F.softmax(p, dim=1)
        q = F.softmax(q, dim=1)
        m = 0.5 * (p + q)
        kl_pm = F.kl_div(torch.log(p + 1e-10), m, reduction='none').sum(dim=1)
        kl_qm = F.kl_div(torch.log(q + 1e-10), m, reduction='none').sum(dim=1)
        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd
    
    def forward(self, logits_str: torch.Tensor, logits_seq: torch.Tensor) -> torch.Tensor:
        jsd = self.js_divergence(logits_str, logits_seq)
        return jsd.mean()


class LearnableLossWeights(nn.Module):
    def __init__(self, init_alpha: float = 0.1, init_beta: float = 0.1, constraint: str = 'softplus'):
        super().__init__()
        self.constraint = constraint
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))
        self.log_beta = nn.Parameter(torch.tensor(math.log(init_beta)))
    
    @property
    def alpha(self) -> torch.Tensor:
        if self.constraint == 'softplus':
            return F.softplus(self.log_alpha)
        elif self.constraint == 'sigmoid':
            return torch.sigmoid(self.log_alpha) * 2
        elif self.constraint == 'exp':
            return torch.exp(self.log_alpha)
        elif self.constraint == 'relu':
            return F.relu(self.log_alpha) + 0.01
        else:
            return torch.exp(self.log_alpha)
    
    @property
    def beta(self) -> torch.Tensor:
        if self.constraint == 'softplus':
            return F.softplus(self.log_beta)
        elif self.constraint == 'sigmoid':
            return torch.sigmoid(self.log_beta) * 2
        elif self.constraint == 'exp':
            return torch.exp(self.log_beta)
        elif self.constraint == 'relu':
            return F.relu(self.log_beta) + 0.01
        else:
            return torch.exp(self.log_beta)
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.alpha, self.beta


class MACLoss(nn.Module):
    def __init__(self, 
                 num_classes: int = 3,
                 init_alpha: float = 0.1,
                 init_beta: float = 0.1,
                 temperature: float = 0.07,
                 gamma: float = 1.0,
                 use_pmcl: bool = True,
                 use_lcon: bool = True,
                 weight_constraint: str = 'softplus',
                 weight_reg_lambda: float = 0.01):
        super().__init__()
        self.num_classes = num_classes
        self.use_pmcl = use_pmcl
        self.use_lcon = use_lcon
        self.weight_reg_lambda = weight_reg_lambda
        
        self.loss_weights = LearnableLossWeights(
            init_alpha=init_alpha,
            init_beta=init_beta,
            constraint=weight_constraint
        )
        
        self.task_loss = nn.CrossEntropyLoss()
        
        if self.use_pmcl:
            self.pmcl_loss = PriorModulatedContrastiveLoss(
                temperature=temperature, gamma=gamma
            )
        
        if self.use_lcon:
            self.lcon_loss = LogicConsistencyConstraint()
            self.aux_classifier_str = nn.Linear(512, num_classes)
            self.aux_classifier_seq = nn.Linear(512, num_classes)
    
    def forward(self,
                fusion_features: torch.Tensor,
                primary_logits: torch.Tensor,
                labels: torch.Tensor,
                GPK_vectors: Optional[torch.Tensor] = None,
                str_features: Optional[torch.Tensor] = None,
                seq_features: Optional[torch.Tensor] = None,
                return_components: bool = False):
        
        alpha, beta = self.loss_weights()
        
        losses = {}
        weight_info = {
            'alpha': alpha.item(),
            'beta': beta.item(),
            'task_weight': 1.0
        }
        
        loss_task = self.task_loss(primary_logits, labels)
        losses['task'] = loss_task
        
        if self.use_pmcl and GPK_vectors is not None:
            loss_pmcl = self.pmcl_loss(fusion_features, GPK_vectors, labels)
            losses['pmcl'] = loss_pmcl
            weight_info['pmcl_weight'] = alpha.item()
        else:
            loss_pmcl = torch.tensor(0.0, device=primary_logits.device)
            losses['pmcl'] = loss_pmcl
            weight_info['pmcl_weight'] = 0.0
        
        if self.use_lcon and str_features is not None and seq_features is not None:
            logits_str = self.aux_classifier_str(str_features)
            logits_seq = self.aux_classifier_seq(seq_features)
            loss_lcon = self.lcon_loss(logits_str, logits_seq)
            losses['lcon'] = loss_lcon
            weight_info['lcon_weight'] = beta.item()
        else:
            loss_lcon = torch.tensor(0.0, device=primary_logits.device)
            losses['lcon'] = loss_lcon
            weight_info['lcon_weight'] = 0.0
        
        total_loss = loss_task + alpha * loss_pmcl + beta * loss_lcon
        losses['total'] = total_loss
        
        if return_components:
            return losses
        else:
            return total_loss


# =========================================================
# 3. Intelligent GPK Data Loader
# =========================================================

class IntelligentGPKDataLoader:
    def __init__(self, csv_path: str, normalize: bool = True, fill_missing: bool = True):
        self.csv_path = Path(csv_path)
        self.data = None
        self.gpk_columns = None
        self.gpk_dim = None
        self.normalize = normalize
        self.fill_missing = fill_missing
        self.statistics = {}
    
    def load_data(self):
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        self.data = pd.read_csv(self.csv_path)
        print(f"[GPK] Loaded data: {len(self.data)} records, {len(self.data.columns)} columns")
        
        # Assume ID column name is 'id', other columns are GPK features
        if 'id' not in self.data.columns:
            # If no 'id' column, assume first column is ID
            id_col = self.data.columns[0]
            self.data = self.data.rename(columns={id_col: 'id'})
        
        self.gpk_columns = [col for col in self.data.columns if col != 'id']
        self.gpk_dim = len(self.gpk_columns)
        
        print(f"[GPK] GPK feature dimension: {self.gpk_dim}")
        print(f"[GPK] GPK features: {self.gpk_columns[:5]}...")
        
        self._preprocess_data()
        self._compute_statistics()
        
        return self
    
    def _preprocess_data(self):
        print(f"[GPK] Data preprocessing...")
        
        if self.fill_missing:
            missing_before = self.data[self.gpk_columns].isnull().sum().sum()
            for col in self.gpk_columns:
                if self.data[col].isnull().any():
                    col_mean = self.data[col].mean()
                    self.data[col] = self.data[col].fillna(col_mean)
            missing_after = self.data[self.gpk_columns].isnull().sum().sum()
            print(f"[GPK] Filled missing values: {missing_before} -> {missing_after}")
        
        if self.normalize:
            for col in self.gpk_columns:
                mean = self.data[col].mean()
                std = self.data[col].std()
                if std > 0:
                    self.data[col] = (self.data[col] - mean) / std
                else:
                    self.data[col] = 0
            print(f"[GPK] Data normalization completed")
    
    def _compute_statistics(self):
        for col in self.gpk_columns:
            self.statistics[col] = {
                'mean': float(self.data[col].mean()),
                'std': float(self.data[col].std()),
                'min': float(self.data[col].min()),
                'max': float(self.data[col].max())
            }
    
    def get_gpk_vector(self, file_id: str) -> np.ndarray:
        possible_ids = self._generate_id_variants(file_id)
        
        for possible_id in possible_ids:
            match = self.data[self.data['id'] == possible_id]
            if len(match) > 0:
                gpk_vector = match[self.gpk_columns].values[0]
                return gpk_vector.astype(np.float32)
        
        print(f"[WARNING] GPK data not found: {file_id}")
        print(f"  Attempted IDs: {possible_ids[:3]}...")
        print(f"  Using zero vector as replacement")
        
        return np.zeros(self.gpk_dim, dtype=np.float32)
    
    def _generate_id_variants(self, file_id: str) -> List[str]:
        variants = []
        variants.append(file_id)
        
        stem = Path(file_id).stem
        
        if stem.endswith('.bases'):
            variants.append(stem.replace('.bases', ''))
        elif stem.endswith('.cigar'):
            variants.append(stem.replace('.cigar', ''))
        else:
            variants.append(stem)
        
        parts = stem.split('.')
        if len(parts) >= 4:
            base_id = '.'.join(parts[:4])
            variants.append(base_id)
            
            if parts[0].startswith('chr'):
                variants.append(parts[0][3:] + '.' + '.'.join(parts[1:4]))
        
        for suffix in ['.bases', '.cigar', '.png', '.jpg', '.jpeg']:
            if stem.endswith(suffix):
                variants.append(stem[:-len(suffix)])
        
        if '_' in stem:
            parts = stem.split('_')
            if len(parts) >= 4:
                variants.append('_'.join(parts[:4]))
        
        unique_variants = []
        seen = set()
        for variant in variants:
            if variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)
        
        return unique_variants
    
    def get_info(self) -> Dict[str, Any]:
        return {
            'num_samples': len(self.data),
            'gpk_dim': self.gpk_dim,
            'columns': self.gpk_columns,
            'statistics': {k: v for k, v in list(self.statistics.items())[:3]}
        }


if __name__ == "__main__":
    print("Adaptive GPK dimension CDCD loss function module loaded successfully")
    print("Common modules removed, imported from modules.py")