#!/usr/bin/env python3
"""
Common Module Definitions File
Contains all shared model components and utility functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
import warnings


# =========================================================
# 1. Basic Model Construction Functions
# =========================================================

def build_backbone_with_layered_finetuning(model_name, num_classes=3, freeze_layers=5, trainable_layers=3):
    """Build backbone with layer-wise fine-tuning"""
    if model_name.startswith('resnet'):
        model = getattr(models, model_name)(pretrained=True)
        
        # ResNet layer structure
        layers = []
        layers.append(model.conv1)
        layers.append(model.bn1)
        layers.append(model.relu)
        layers.append(model.maxpool)
        layers.extend(list(model.layer1))
        layers.extend(list(model.layer2))
        layers.extend(list(model.layer3))
        layers.extend(list(model.layer4))
        
        # Freeze bottom layers
        for i, layer in enumerate(layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Remove classification head
        model.fc = nn.Identity()
        
        # Get feature dimension
        def get_feature_dim(x):
            with torch.no_grad():
                x = model.conv1(x)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = model.avgpool(x)
                x = torch.flatten(x, 1)
                return x.shape[-1]
        
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        
        layers = list(model.features.children())
        
        for i, layer in enumerate(layers):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
        
        model.classifier = nn.Identity()
        
        def get_feature_dim(x):
            with torch.no_grad():
                x = model.features(x)
                x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
                x = torch.flatten(x, 1)
                return x.shape[-1]
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model.get_feature_dim = get_feature_dim
    return model


# =========================================================
# 2. Feature Processing Modules
# =========================================================

class AdaptiveFeatureProjection(nn.Module):
    """Adaptive feature projection"""
    def __init__(self, in_dim1, in_dim2, out_dim=512):
        super().__init__()
        self.projection1 = nn.Linear(in_dim1, out_dim)
        self.projection2 = nn.Linear(in_dim2, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, feat1, feat2):
        feat1_proj = self.relu(self.norm1(self.projection1(feat1)))
        feat2_proj = self.relu(self.norm2(self.projection2(feat2)))
        return feat1_proj, feat2_proj


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion"""
    def __init__(self, feature_dim):
        super().__init__()
        self.query1 = nn.Linear(feature_dim, feature_dim)
        self.key1 = nn.Linear(feature_dim, feature_dim)
        self.value1 = nn.Linear(feature_dim, feature_dim)
        
        self.query2 = nn.Linear(feature_dim, feature_dim)
        self.key2 = nn.Linear(feature_dim, feature_dim)
        self.value2 = nn.Linear(feature_dim, feature_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
    
    def forward(self, feat1, feat2):
        feat1 = feat1.unsqueeze(1)
        feat2 = feat2.unsqueeze(1)
        
        q1 = self.query1(feat1)
        k1 = self.key1(feat1)
        v1 = self.value1(feat1)
        
        q2 = self.query2(feat2)
        k2 = self.key2(feat2)
        v2 = self.value2(feat2)
        
        attn1 = self.softmax(torch.bmm(q1, k2.transpose(1, 2)))
        out1 = torch.bmm(attn1, v2)
        
        attn2 = self.softmax(torch.bmm(q2, k1.transpose(1, 2)))
        out2 = torch.bmm(attn2, v1)
        
        out1 = self.gamma1 * out1 + feat1
        out2 = self.gamma2 * out2 + feat2
        
        fused_feat = torch.cat([out1, out2], dim=-1)
        fused_feat = fused_feat.squeeze(1)
        
        return fused_feat


# =========================================================
# 3. CDCD Core Modules
# =========================================================

class ConfidenceWeightedCrossAttention(nn.Module):
    """Confidence-weighted cross-attention"""
    def __init__(self, feature_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert self.head_dim * num_heads == feature_dim, f"feature_dim {feature_dim} must be divisible by num_heads {num_heads}"
        
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, query_features, key_features, value_features, confidence_scores=None):
        batch_size, seq_len, _ = query_features.shape
        
        Q = self.query_proj(query_features)
        K = self.key_proj(key_features)
        V = self.value_proj(value_features)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if confidence_scores is not None:
            if confidence_scores.dim() == 1:
                confidence_scores = confidence_scores.view(batch_size, 1, 1, 1)
            elif confidence_scores.dim() == 2:
                confidence_scores = confidence_scores.view(batch_size, 1, seq_len, 1)
            K = K * confidence_scores
            V = V * confidence_scores
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        context = torch.matmul(attention_probs, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        output = self.output_proj(context)
        output = self.output_dropout(output)
        refined_features = output + query_features
        
        return refined_features


class AdaptiveConfidenceMapEstimation(nn.Module):
    """Adaptive GPK dimension confidence map estimation"""
    def __init__(self, feature_channels: int, gpk_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.gpk_dim = gpk_dim
        self.feature_channels = feature_channels
        
        # Automatically adjust hidden layer size based on GPK dimension
        if gpk_dim < 20:
            gpk_hidden1 = max(32, gpk_dim * 2)
            gpk_hidden2 = max(16, gpk_dim)
            mlp_hidden1 = min(hidden_dim, 64)
            mlp_hidden2 = mlp_hidden1 // 2
        elif gpk_dim < 50:
            gpk_hidden1 = max(64, gpk_dim)
            gpk_hidden2 = max(32, gpk_dim // 2)
            mlp_hidden1 = min(hidden_dim, 128)
            mlp_hidden2 = mlp_hidden1 // 2
        else:
            gpk_hidden1 = min(256, max(128, gpk_dim // 2))
            gpk_hidden2 = min(128, max(64, gpk_dim // 4))
            mlp_hidden1 = hidden_dim
            mlp_hidden2 = hidden_dim // 2
        
        if gpk_dim > 200:
            warnings.warn(f"GPK dimension is large ({gpk_dim}), consider feature selection or dimensionality reduction")
        
        # Global prior knowledge encoder
        self.gpk_encoder = nn.Sequential(
            nn.Linear(gpk_dim, gpk_hidden1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(gpk_hidden1, gpk_hidden2),
            nn.ReLU(),
            nn.Dropout(0.1) if gpk_dim > 30 else nn.Identity(),
            nn.Linear(gpk_hidden2, mlp_hidden2),
            nn.ReLU()
        )
        
        # Confidence estimation MLP
        self.confidence_mlp = nn.Sequential(
            nn.Linear(feature_channels + mlp_hidden2, mlp_hidden1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden1, mlp_hidden2),
            nn.ReLU(),
            nn.Dropout(0.1) if gpk_dim > 30 else nn.Identity(),
            nn.Linear(mlp_hidden2, 1),
            nn.Sigmoid()
        )
        
        self.config = {
            'gpk_dim': gpk_dim,
            'feature_channels': feature_channels,
            'gpk_hidden1': gpk_hidden1,
            'gpk_hidden2': gpk_hidden2,
            'mlp_hidden1': mlp_hidden1,
            'mlp_hidden2': mlp_hidden2
        }
    
    def forward(self, feature_map: torch.Tensor, gpk_vector: torch.Tensor) -> torch.Tensor:
        batch_size = feature_map.size(0)
        
        # Dimension check and automatic adjustment
        if gpk_vector.size(1) != self.gpk_dim:
            gpk_vector = self._adjust_gpk_dimension(gpk_vector, batch_size)
        
        # Global average pooling
        pooled_features = F.adaptive_avg_pool2d(feature_map, (1, 1))
        pooled_features = pooled_features.view(batch_size, -1)
        
        # Encode global prior knowledge
        gpk_encoded = self.gpk_encoder(gpk_vector)
        
        # Feature concatenation and confidence estimation
        combined_features = torch.cat([pooled_features, gpk_encoded], dim=1)
        confidence_score = self.confidence_mlp(combined_features)
        
        return confidence_score.squeeze(1)
    
    def _adjust_gpk_dimension(self, gpk_vector: torch.Tensor, batch_size: int) -> torch.Tensor:
        actual_dim = gpk_vector.size(1)
        
        if actual_dim < self.gpk_dim:
            padding = torch.zeros(batch_size, self.gpk_dim - actual_dim, 
                                device=gpk_vector.device)
            adjusted_vector = torch.cat([gpk_vector, padding], dim=1)
            if self.training:
                warnings.warn(f"GPK dimension insufficient: expected {self.gpk_dim}, got {actual_dim}, padded with zeros")
        elif actual_dim > self.gpk_dim:
            adjusted_vector = gpk_vector[:, :self.gpk_dim]
            if self.training:
                warnings.warn(f"GPK dimension excessive: expected {self.gpk_dim}, got {actual_dim}, truncated")
        else:
            adjusted_vector = gpk_vector
        
        return adjusted_vector
    
    def get_config(self) -> Dict[str, Any]:
        return self.config


class AdaptiveCDCDModule(nn.Module):
    """Adaptive GPK dimension CDCD module"""
    def __init__(self, 
                 feature_dim: int,
                 gpk_dim: int,
                 num_heads: int = 8,
                 hidden_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.gpk_dim = gpk_dim
        
        # Automatically adjust number of attention heads based on feature dimension
        if feature_dim % 8 != 0:
            suggested_num_heads = self._find_optimal_num_heads(feature_dim)
            if suggested_num_heads != num_heads:
                print(f"[INFO] Adjusting number of attention heads: {num_heads} -> {suggested_num_heads} (feature dimension: {feature_dim})")
                num_heads = suggested_num_heads
        
        # Adaptive CME modules
        self.cme_str = AdaptiveConfidenceMapEstimation(feature_dim, gpk_dim, hidden_dim)
        self.cme_seq = AdaptiveConfidenceMapEstimation(feature_dim, gpk_dim, hidden_dim)
        
        # CWCA modules
        self.cwca_str_to_seq = ConfidenceWeightedCrossAttention(feature_dim, num_heads, dropout)
        self.cwca_seq_to_str = ConfidenceWeightedCrossAttention(feature_dim, num_heads, dropout)
        
        # Feature normalization
        self.layer_norm_str = nn.LayerNorm(feature_dim)
        self.layer_norm_seq = nn.LayerNorm(feature_dim)
        
        # Adaptive FFN
        self.ffn = self._build_adaptive_ffn(feature_dim, hidden_dim, dropout)
        
        print(f"[CDCD] Configuration: feature_dim={feature_dim}, gpk_dim={gpk_dim}, "
              f"num_heads={num_heads}, hidden_dim={hidden_dim}")
    
    def _find_optimal_num_heads(self, feature_dim: int) -> int:
        for num_heads in [8, 4, 2, 1]:
            if feature_dim % num_heads == 0:
                return num_heads
        return 1
    
    def _build_adaptive_ffn(self, feature_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
        if feature_dim <= 128:
            return nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim * 4, feature_dim * 2),
                nn.ReLU(),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.LayerNorm(feature_dim)
            )
        elif feature_dim <= 512:
            return nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim * 4, feature_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.LayerNorm(feature_dim)
            )
        else:
            return nn.Sequential(
                nn.Linear(feature_dim * 2, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, feature_dim),
                nn.LayerNorm(feature_dim)
            )
    
    def forward(self, 
                H_str: torch.Tensor,
                H_seq: torch.Tensor,
                gpk_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size, seq_len, feature_dim = H_str.shape
        
        # Calculate confidence maps
        H_str_img = H_str.transpose(1, 2).unsqueeze(2)
        H_seq_img = H_seq.transpose(1, 2).unsqueeze(2)
        
        confidence_str = self.cme_str(H_str_img, gpk_vector)
        confidence_seq = self.cme_seq(H_seq_img, gpk_vector)
        
        # Feature normalization
        H_str_norm = self.layer_norm_str(H_str)
        H_seq_norm = self.layer_norm_seq(H_seq)
        
        # Bidirectional confidence-weighted cross-attention
        H_seq_refined = self.cwca_str_to_seq(
            query_features=H_str_norm,
            key_features=H_seq_norm,
            value_features=H_seq_norm,
            confidence_scores=confidence_seq.view(batch_size, 1).expand(-1, seq_len)
        )
        
        H_str_refined = self.cwca_seq_to_str(
            query_features=H_seq_norm,
            key_features=H_str_norm,
            value_features=H_str_norm,
            confidence_scores=confidence_str.view(batch_size, 1).expand(-1, seq_len)
        )
        
        # Generate joint representation
        combined_features = torch.cat([H_str_refined, H_seq_refined], dim=-1)
        H_fusion = self.ffn(combined_features)
        
        return H_fusion, confidence_str, confidence_seq


# =========================================================
# 4. Chromosome Name Processing Utility Functions
# =========================================================

def normalize_chromosome_name(chr_str):
    """Normalize chromosome name"""
    if not chr_str:
        return None
    
    chr_str = str(chr_str).strip()
    
    if chr_str.isdigit():
        chr_num = int(chr_str)
        if 1 <= chr_num <= 22:
            return f"chr_{chr_num}"
        else:
            return None
    
    chr_str = chr_str.lower()
    chr_str = chr_str.replace('chromosome', '').replace('chrom', '').replace('chr', '')
    chr_str = chr_str.strip('_').strip()
    
    valid_chromosomes = {
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
        '11', '12', '13', '14', '15', '16', '17', '18', '19',
        '20', '21', '22', 'x', 'y'
    }
    
    if chr_str in valid_chromosomes:
        return f"chr_{chr_str.upper()}"
    
    patterns = [
        r'^chr_?([0-9xy]+)$',
        r'^([0-9xy]+)$',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, chr_str)
        if match:
            chr_num = match.group(1).upper()
            if chr_num in ['X', 'Y'] or chr_num.isdigit():
                return f"chr_{chr_num}"
    
    return None


def extract_chr_from_filename(name):
    """Extract chromosome information from filename"""
    name = Path(name).stem
    
    patterns = [
        r'^([0-9XY]+)\.[0-9]+\.[A-Z]+\.[0-9]+',
        r'chr_?([0-9XY]+)\.[0-9]+\.[A-Z]+\.[0-9]+',
        r'([0-9XY]+)_[0-9]+_[A-Z]+_[0-9]+',
    ]
    
    for pattern in patterns:
        match = re.match(pattern, name, re.IGNORECASE)
        if match:
            chr_str = match.group(1)
            normalized = normalize_chromosome_name(chr_str)
            if normalized:
                return normalized
    
    for token in name.split('.'):
        normalized = normalize_chromosome_name(token)
        if normalized:
            return normalized
    
    for token in name.split('_'):
        normalized = normalize_chromosome_name(token)
        if normalized:
            return normalized
    
    return None


def extract_id_from_filename(filename):
    """Extract unique identifier from filename"""
    name = Path(filename).stem
    
    patterns = [
        r'([0-9XY]+)\.[0-9]+\.[A-Z]+\.[0-9]+',
        r'chr_?([0-9XY]+)\.[0-9]+\.[A-Z]+\.[0-9]+',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            full_match = match.group(0)
            if full_match.startswith('chr_'):
                full_match = full_match[4:]
            elif full_match.startswith('chr'):
                full_match = full_match[3:]
            return full_match
    
    parts = name.split('.')
    if len(parts) >= 4:
        id_parts = parts[:4]
        if id_parts[-1] in ['bases', 'cigar']:
            id_parts = parts[:4]
        return '.'.join(id_parts)
    
    parts = name.split('_')
    if len(parts) >= 4:
        return '_'.join(parts[:4])
    
    return name.replace('.bases', '').replace('.cigar', '')


if __name__ == "__main__":
    print("Common module definitions file loaded successfully")