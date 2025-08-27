# AIGCMatch for Medical Image Segmentation


## Results




## Getting Started

### Installation

```bash
cd UniMatch
conda create -n unimatch python=3.10.4
conda activate unimatch
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


## Usage

### AIGCMatch

```bash
# use torch.distributed.launch
# switch to current folder
sh scripts/AIGCMATCH.sh <num_gpu> <port>
```




