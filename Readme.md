&nbsp;

# WDCFNet: A Wavelet-Guided Dual-Branch Cross-Attention Fusion Network for Low-Light Image Enhancement

Shuqin Zhang, Kaiyu Li, Xueqing Wang, Yuanqing Xia






## Proposed WDCFNet 

<details close>
<summary><b>WDCFNet Graphical Abstract:</b></summary>

![results0](.\images\graphicalabstract.png)

</details>




## Visual Comparison  
<details close>
<summary><b>LOL-v1, LOL-v2-real, and LOL-v2-synthetic:</b></summary>
    
![results1](./images/LOLv1.png)

![results2](./images/LOLv2.png)

</details>

<details close>
<summary><b>LSRW-Huawei and LSRW-Nikon:</b></summary>


![results3](./images/HuaweiNikon.png)

</details>

<details close>
<summary><b>DICM, LIME, MEF, and VV:</b></summary>


![results4](./images/unpair.png)

</details>

## Weights and Results 
All the weights that we trained on different datasets is available at [[Baidu Pan](https://pan.baidu.com/s/1sJIkX2-4OU_gfcGmlhIREQ?pwd=YT42)] (code: `YT42`) .  

<details close>
<summary><b>The metrics of WDCFNet on paired datasets are shown in the following table: </b></summary>

| Folder (test datasets)                        | PSNR        | SSIM       | LPIPS      |
| --------------------------------------------- | ----------- | ---------- | ---------- |
| LOLv1                  | 24.57 | 0.8641 | 0.0394 |
| LOLv2-real | 24.42  | 0.8721    | 0.0545 |
| LOLv2-syn | 26.41 | 0.9414 | 0.0203 |
| LSRW-Huawei | 21.31 | 0.6352 | 0.1557 |
| LSRW-Nikon | 17.67 | 0.5194 | 0.2003 |

</details>

<details close>
<summary><b>Performance on four unpaired datasets are shown in the following table:  </b></summary>


| metrics | DICM | LIME | MEF  | VV   |
| ------- | ---- | ---- | ---- | ---- |
| NIQE    | 3.75 | 4.04 | 3.71 | 3.76 |

</details>

## 1. Get Started 

### Dependencies and Installation

- Python 3.7.12
- Pytorch 1.8.1

(1) Create Conda Environment

```bash
conda create --name WDCFNet python=3.7.12
conda activate WDCFNet
```

(2) Clone Repo

```bash
git clone git@github.com:mlxgccc/WDCFNet.git
```

(3) Install Dependencies

```bash
cd WDCFNet
pip install -r requirements.txt
```


### Data Preparation

You can refer to the following links to download the datasets. 

- All datasets: [Baidu Pan](https://pan.baidu.com/s/1SwbzKsGpXSFLSYN_7-DQOQ?pwd=YT42) (code: `YT42`).

</details>

## 2. Testing 


Download our weights from [[Baidu Pan](https://pan.baidu.com/s/1sJIkX2-4OU_gfcGmlhIREQ?pwd=YT42)] (code: `YT42`) and put them in folder `weights`:

<details close> <summary>weights (click to expand)</summary>

```
├── weights
    ├── LOL.pth
    ├── LOLv2real.pth
    ├── LOLv2Syn.pth
    ├── huawei.pth
    ├── nikon.pth
    ├── huawei_LIME.pth
    ├── huaweiDICM.pth
    ├── HuaweiMEF.pth
    ├── HuaweiVV.pth
```

</details>



You can test our method as followed, all the results will saved in `./output` folder:

<details close> <summary>(click to expand)</summary>

```bash
# LOLv1
python eval.py --lol 

# LOLv2-real
python eval.py --lol_v2_real 

# LOLv2-syn
python eval.py --lol_v2_syn 

# LSRW-Huawei
python eval.py --huawei

# LSRW-Nikon
python eval.py --nikon

# DICM
python eval.py --DICM

# LIME
python eval.py --LIME

# MEF
python eval.py --MEF

# VV
python eval.py --VV
```

</details>

- Also, you can test all the metrics mentioned in our paper as follows:
  
<details close> <summary>(click to expand)</summary>

```bash
# LOLv1
python measure.py --lol

# LOLv2-real
python measure.py --lol_v2_real

# LOLv2-syn
python measure.py --lol_v2_syn

# LSRW-Huawei
python measure.py --huawei

# LSRW-Nikon
python measure.py --nikon


# four unpaired datasets DICM, LIME, MEF, VV. 
# You can change "--DICM" to the other unpaired datasets "LIME, MEF, VV".
python measure_niqe_bris.py --DICM

```

</details>


## 3. Training 

- We put all the configurations that need to be adjusted in the `./data/options.py` folder and explained them in the file. 

```bash
python train.py --dataset lol_v1
```

## 4. Contacts 

If you have any questions, please contact us or submit an issue to the repository!

Kaiyu Li (likaiyu@zut.edu.cn)

## 5. Acknowledgments

We thank the following article and the authors for their open-source codes.

```
@article{yan2025hvi,
  title={HVI: A New color space for Low-light Image Enhancement},
  author={Yan, Qingsen and Feng, Yixu and Zhang, Cheng and Pang, Guansong and Shi, Kangbiao and Wu, Peng and Dong, Wei and Sun, Jinqiu and Zhang, Yanning},
  journal={arXiv preprint arXiv:2502.20272},
  year={2025}
}

@misc{feng2024hvi,
      title={You Only Need One Color Space: An Efficient Network for Low-light Image Enhancement}, 
      author={Yixu Feng and Cheng Zhang and Pei Wang and Peng Wu and Qingsen Yan and Yanning Zhang},
      year={2024},
      eprint={2402.05809},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
