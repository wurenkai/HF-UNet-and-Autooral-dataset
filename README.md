<p align="center">
  <h1 align="center">A high-order focus interaction model and oral ulcer dataset for oral ulcer segmentation</h1>
  <p align="center">
    Chenghao Jiang, Renkai Wu, Yinghao Liu, Yue Wang, Qing Chang, Pengchen Liang*, and Yuan Fan*
  </p>
    <p align="center">
      1. Nanjing Medical University, Nanjing, China</br>
      2. Shanghai University, Shanghai, China</br>
      3. The Affiliated Stomatological Hospital of Nanjing Medical University, Nanjing, China</br>
      4. Jiangsu Province Engineering Research Center of Stomatological Translational Medicine, Nanjing, China</br>
      5. University of Shanghai for Science and Technology, Shanghai, China</br>
  </p>
</p>

## Examples of proposed dataset segmentation tasks

https://github.com/wurenkai/HF-UNet-and-Autooral-dataset/assets/124028634/de265270-450c-4395-bb97-0cf9d048c601


## Examples of proposed dataset classification tasks

https://github.com/wurenkai/HF-UNet-and-Autooral-dataset/assets/124028634/94c56a9f-0130-4a96-8cfa-a2e5048e504b

**0. Main Environments.**
- python 3.8
- pytorch 1.12.0

**1. The proposed datasets (Autooral dataset).** </br>
(1) The Autooral dataset is available [here](https://drive.google.com/file/d/1n29L25N4H0XFfyWxle95PqS6lyQwNv64/view?usp=sharing). It should be noted:
1. If you use the dataset, please cite the paper: https://www.nature.com/articles/s41598-024-69125-9 
2. The Autooral dataset may only be used for academic research, not for commercial purposes.
3. If you can, please give us a like (Starred) for our GitHub project: https://github.com/wurenkai/HF-UNet-and-Autooral-dataset

(2) After getting the Autooral dataset, execute 'Prepare_Autooral.py' for preprocessing to generate the npy file. We also provide annotations for categorization to provide more richness to the study. </br>

**2. Train the HF-UNet.** </br>
Modify the dataset address in the config_setting.py file to the address where the npy is stored after preprocessing. Then, perform the following operation:
```
python train.py
```
- After trianing, you could obtain the outputs in './results/'

**3. Test the HF-UNet.** </br>
First, in the test.py file, you should change the address of the checkpoint in 'resume_model' and fill in the location of the test data in 'data_path'.
```
python test.py
```
- After testing, you could obtain the outputs in './results/'

## Citation
If you find this repository helpful, please consider citing:
```
@article{jiang2024high,
  title={A high-order focus interaction model and oral ulcer dataset for oral ulcer segmentation},
  author={Jiang, Chenghao and Wu, Renkai and Liu, Yinghao and Wang, Yue and Chang, Qing and Liang, Pengchen and Fan, Yuan},
  journal={Scientific Reports},
  volume={14},
  number={1},
  pages={20085},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```

## References
- [MHorUNet](https://github.com/wurenkai/MHorUNet)
- [HorNet](https://github.com/raoyongming/HorNet)
---
