## Introduction
This is the source code of [Cephalometric Landmark Detection by Attentive Feature Pyramid Fusion and Regression-Voting](https://arxiv.org/pdf/1908.08841.pdf). The paper is early accepted in MICCAI 2019.



## Prerequistes
- Python 3.8
- PyTorch 1.0.0-1.7.0

## Dataset and setup
- Download the dataset from the official [webside](https://figshare.com/s/37ec464af8e81ae6ebbf) and put it in the root. We also provide the [processed dataset](https://connecthkuhk-my.sharepoint.com/personal/crnsmile_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fcrnsmile%5Fconnect%5Fhku%5Fhk%2FDocuments%2FGoogle%5Fdrive%2Fprocess%5Fdata%2Ezip&parent=%2Fpersonal%2Fcrnsmile%5Fconnect%5Fhku%5Fhk%2FDocuments%2FGoogle%5Fdrive&fromShare=true&ga=1) for a quick start.

## Training and validation
- python main.py

## Reference

If you found this code useful, please cite the following paper:

```
@inproceedings{chen2019cephalometric,
  title={Cephalometric landmark detection by attentive feature pyramid fusion and regression-voting},
  author={Chen, Runnan and Ma, Yuexin and Chen, Nenglun and Lee, Daniel and Wang, Wenping},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={873--881},
  year={2019},
  organization={Springer}
}
```
