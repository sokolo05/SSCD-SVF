# SSCD-SVF: Semantic-Syntactic Collaborative Denoising: A Knowledge-Aware Multimodal Framework for Genomic Structural Variant Filtering

<img width="1306" height="769" alt="image" src="https://github.com/sokolo05/CoDAC/blob/main/image/main-2.png" />

<div align="center">
**Overall architecture of SSCD-SVF with confidence-driven collaborative denoising and multi-level knowledge-aware constraints.**

</div>

# 📖 Overview

Semantic-Syntactic Collaborative Denoising for SV Filtering (**SSCD-SVF**) integrates nucleotide-level sequence semantics and alignment-derived syntactic structures through a knowledge-guided dual-stream architecture. Genomic priors (e.g., mapping quality, GC content, repeat annotations) dynamically regulate cross-modal fusion via a confidence-gated mechanism, while multi-level constraints ensure biologically consistent predictions. The framework achieves robust SV filtering in noisy long-read data.

# ✨ Key Features

🔀 Dual-Modal Architecture: Integrates syntactic (CIGAR) and semantic (bases) features from genomic data

🎯 Confidence-Driven Denoising: Adaptive confidence estimation with cross-modal correction

📊 Multi-level knowledge-aware constraints: Feature-space contrastive learning and decision consistency regularization

🚀 Distributed Training: Native PyTorch DDP support for multi-GPU training

🔄 Adaptive GPK Dimensions: Intelligent handling of varying global prior knowledge dimensions

📈 State-of-the-art Performance: Superior filtering accuracy across multiple sequencing platforms

# 🚀 Quick Start

## Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 4+ GB VRAM per GPU recommended
- 16+ GB RAM

## Installation

```
# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pysam==0.23.3 pandas numpy matplotlib scikit-learn scipy

# Install specialized bioinformatics packages
pip install biopython mappy cuteSV sniffles intervaltree pyfaidx PyVCF3 Truvari Badread

# Install packages from requirements.txt
pip install -r requirements.txt
```

## Essential Bioinformatics Dependencies

| Package                                                      | Purpose                             |
| ------------------------------------------------------------ | ----------------------------------- |
| ![pysam](https://img.shields.io/badge/pysam-0.23.3-14a2b8?logo=python&logoColor=white) | BAM/CRAM file processing            |
| ![biopython](https://img.shields.io/badge/biopython-1.85-14a2b8?logo=python&logoColor=white) | Sequence analysis and manipulation  |
| ![mappy](https://img.shields.io/badge/mappy-2.30-14a2b8?logo=python&logoColor=white) | Sequence alignment and mapping      |
| ![cuteSV](https://img.shields.io/badge/cuteSV-2.1.3-14a2b8?logo=python&logoColor=white) | SV detection and calling            |
| ![sniffles](https://img.shields.io/badge/sniffles-2.6.3-14a2b8?logo=python&logoColor=white) | Long-read SV caller                 |
| ![intervaltree](https://img.shields.io/badge/intervaltree-3.1.0-14a2b8?logo=python&logoColor=white) | Genomic interval handling           |
| ![pyfaidx](https://img.shields.io/badge/pyfaidx-0.9.0.3-14a2b8?logo=python&logoColor=white) | FASTA file indexing and access      |
| ![PyVCF3](https://img.shields.io/badge/PyVCF3-1.0.4-14a2b8?logo=python&logoColor=white) | VCF file parsing and writing        |
| ![Truvari](https://img.shields.io/badge/Truvari-3.5.0-14a2b8?logo=python&logoColor=white) | SV benchmarking and comparison      |
| ![Badread](https://img.shields.io/badge/Badread-0.4.0-14a2b8?logo=python&logoColor=white) | Read simulation and quality control |

# 📁 Data Preparation Pipeline

## Generate SV Images

```
python generate_images.py \
  --txt_file /path/to/HG002_SVs_Tier1_v0.6.PASS.ALL.pos.txt \
  --bam_file /path/to/HG002.CLR.70x.bam \
  --ref_file /path/to/human_hs37d5.fasta \
  --output_dir ./01.sv_images \
  --extend_length 500 \
  --select_read 70 \
  --csv_out sv_statistics.csv

After generate_images.py:
01.sv_images/
├── bases/*.png               # Semantic images
└── cigar/*.png               # Syntactic images
```

## Generate GPK

```
python generate_gpk.py \
  --txt_file /home/laicx/02.study/03.bib/01.data_process/HG002_SVs_Tier1_v0.6.PASS.ALL.pos.txt \
  --bam_file /home/ifs/laicx/00.dataset/01.bam_file/02.HG002_bam/HG002.CSS.28x.bam \
  --ref_file /home/ifs/laicx/00.dataset/03.ref_file/01.hg37/human_hs37d5.fasta \
  --output_dir /home/laicx/02.study/03.bib/02.get_data/01.sv_images/02.CSS28 \
  --extend_length 500 \
  --select_read 28\
  --csv_out sv_gpk.csv

After generate_gpk.py:
./
└── sv_gpk.csv               # Global prior knowledge
```

## Organize Data by SV Type

```
python split_data.py \
  ./01.sv_images \
  ./02.split_data

02.split_data/
├── bases/
│   ├── Del_positive/*.png
│   ├── Ins_positive/*.png
│   └── Match_negative/*.png
└── cigar/
    ├── Del_positive/*.png
    ├── Ins_positive/*.png
    └── Match_negative/*.png
```

# 🏋️ Training

## Single GPU Training

```
python CoDAC-main.py \
  --bases_root ./02.split_data/bases \
  --cigar_root ./02.split_data/cigar \
  --gpk_csv ./01.sv_images/sv_statistics.csv \
  --class_dirs Del_positive Ins_positive Match_negative \
  --bases_model resnet50 \
  --cigar_model mobilenet_v2 \
  --train_chrs 1 2 \
  --test_chrs 13 \
  --use_cdcd \
  --use_mac \
  --epochs 30 \
  --batch_size 64 \
  --save_path ./dual_icme.pth \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --early_stop_patience 5
```

## Multi-GPU Distributed Training (Recommended)

```
torchrun --nproc_per_node=4 CoDAC-main.py \
  --bases_root ./02.split_data/bases \
  --cigar_root ./02.split_data/cigar \
  --gpk_csv ./01.sv_images/sv_statistics.csv \
  --class_dirs Del_positive Ins_positive Match_negative \
  --bases_model resnet50 \
  --cigar_model mobilenet_v2 \
  --train_chrs 1 2 3 4 5 6 7 8 9 10 11 12 \
  --test_chrs 13 14 15 16 17 18 19 20 21 22 X Y \
  --use_cdcd \
  --use_mac \
  --epochs 30 \
  --batch_size 64 \
  --save_path ./dual_icme.pth \
  --lr 1e-4 \
  --weight_decay 1e-2 \
  --early_stop_patience 5
```

## Key Training Parameters:

| Parameter       | Description                          | Default      |
| --------------- | ------------------------------------ | ------------ |
| `--bases_model` | Backbone for semantic modality       | resnet50     |
| `--cigar_model` | Backbone for syntactic modality      | mobilenet_v2 |
| `--use_cdcd`    | Enable collaborative denoising       | True         |
| `--use_mac`     | Enable multi-granularity constraints | True         |
| `--train_chrs`  | Chromosomes for training             | 1-12         |
| `--test_chrs`   | Chromosomes for testing              | 13-22,X,Y    |
| `--batch_size`  | Per-GPU batch size                   | 64           |

# 🏗️ Architecture Overview

```
Input:
├── Semantic Modality (BASES) → ResNet50 → Feature Projection
└── Syntactic Modality (CIGAR) → MobileNetV2 → Feature Projection

Core:
├── Confidence-Driven Collaborative Denoising (CDCD)
│   ├── Confidence Map Estimation (CME)
│   ├── Confidence-Gated Cross-Attention (CGCA)
│   └── Adaptive Residual Correction
└── Multi-Granularity Awareness Constraints (MAC)
    ├── Prior-Modulated Contrastive Loss (PMCL)
    ├── Biological Consistency Regularization (BCR)
    └── Learnable Weight Balancing

Output: SV Classification (MATCH/DEL/INS)
```

## Core Components

### 1. Dual-Stream Feature Extraction

- **Syntactic Modality (CIGAR)**  
  MobileNetV2 for efficient alignment feature extraction  
- **Semantic Modality (BASES)**  
  ResNet50 for deep sequence context understanding  

### 2. Confidence-Driven Collaborative Denoising (CDCD)

```text
Confidence Map Estimation → Confidence-Gated Cross-Attention → Adaptive Residual Correction
```

### 3.Multi-level Adaptive constraints (MAC)

- Prior-Modulated Contrastive Loss (PMCL)
- Biological Consistency Regularization (BCR)
- Adaptive weight balancing with learnable coefficients

