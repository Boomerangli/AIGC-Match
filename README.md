# 🩺 AIGCMatch: Attention-Guided Weak-to-Strong Consistency for Semi-Supervised Medical Image Segmentation

This project introduces **AIGCMatch**, a simple yet efficient semi-supervised learning framework for **medical image segmentation**.
 It leverages **attention-guided perturbation strategies** at both the image and feature levels to achieve **weak-to-strong consistency**, improving segmentation accuracy with limited labeled data.

------

## 🚀 Features

```markdown
- 🎯 **Attention-Guided CutMix (AttCutMix)**:
   Image-level perturbation guided by model attention to preserve semantic integrity.
- 🧠 **Attention-Guided Feature Perturbation (AttFeaPerb)**:
   Feature-level perturbation across weak/strong channels for robust representations.
- 🔗 **Weak-to-Strong Consistency Regularization**:
   Enforces prediction consistency under multi-level perturbations.
- 📉 **Reduced annotation needs**:
   Achieves SOTA with only 1–10% labeled data.
```

## 🧪 Experimental Results 

### ACDC

| Method               | 1 case Dice ↑ | 3 cases Dice ↑ | 7 cases Dice ↑ |
| -------------------- | ------------- | -------------- | -------------- |
| SupBaseline          | 28.50         | 47.83          | 79.41          |
| FixMatch             | 69.67         | 83.12          | 88.31          |
| UniMatch             | 85.43         | 88.86          | 89.85          |
| DiffRect             | 82.40         | 86.95          | 90.18          |
| **AIGCMatch (Ours)** | **86.25**     | **89.53**      | **90.40**      |

- 🏆 **Best Dice**: 90.40% (7 cases)
- ⚡ Outperforms prior SOTA with fewer labeled samples
- 📊 Also improves Jaccard, 95HD, and ASD metrics

------

## 📦 Installation

Requirements:

- Python 3.10+
- PyTorch ≥ 1.12
- GPU with 12GB+ memory

```bash
git clone https://github.com/yourname/AIGCMatch.git
cd AIGCMatch
conda create -n AIGCimatch python=3.10.4
conda activate AIGCimatch
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

### Dataset

- ACDC: [image and mask](https://drive.google.com/file/d/1LOust-JfKTDsFnvaidFOcAVhkZrgW1Ac/view?usp=sharing)

Please modify your dataset path in configuration files.

```
├── [Your ACDC Path]
    └── data
```

## ⚙️ Usage

```bash
sh scripts/AIGCMATCH.sh <num_gpu> <port>
```

------

## 📊 Ablation Studies

| Perturbation Strategy             | Mean Dice (%) |
| --------------------------------- | ------------- |
| Random + Random                   | 89.85         |
| Random + AttFeaPerb               | 90.20         |
| AttCutMix + Random                | 89.92         |
| **AttCutMix + AttFeaPerb (Ours)** | **90.40**     |

✅ Both AttCutMix and AttFeaPerb improve performance, and their combination achieves the best results.

------

## 📚 Citation

If you use AIGCMatch in your research, please cite:

```bibtex
@article{AIGCMatch2025,
  title={Mind the Context: Attention-Guided Weak-to-Strong Consistency for Enhanced Semi-Supervised Medical Image Segmentation},
  author={Your Name},
  year={2025}
}
```

------

## 👨‍💻 Authors

```
Yuxuan Cheng*, Chenxi Shao*,Li Sha,Jie Ma, Yunfei Xie, Guoliang Li
Research Interests: Medical Imaging, Semi-Supervised Learning
```




