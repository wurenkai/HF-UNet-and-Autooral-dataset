<p align="center">
  <h1 align="center">HF-UNet for Autooral Segmentation: High-order Focus Interaction Model and Multi-tasking Oral Ulcer Dataset</h1>
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
(1) To obtain the Autooral dataset, you need to provide **your name, organization, a brief description of your upcoming project, and an assurance that you will not share the dataset privately**. Specifically, complete the information in the following format and send it to 'wurk@shu.edu.cn' with the subject name '**Autooral dataset request**'. We will usually review your request and provide you with a link within 3 days. If you do not register your information as required, your application may fail. Please understand! </br>
```
Name:
Affiliation:
Reason for applying (one sentence description of your work):
I (the applicant) guarantee that the data will be used only for academic communication and not for any commercial purposes. I (the applicant) guarantee that I will not privately disseminate the data to any public place without the consent of the author.
```

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

## References
- [MHorUNet](https://github.com/wurenkai/MHorUNet)
- [HorNet](https://github.com/raoyongming/HorNet)
---
