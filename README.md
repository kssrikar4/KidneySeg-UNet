# KiTS Kidney Tumor Segmentation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.0+-green.svg)](https://monai.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/🤗_Models-HF-yellow.svg)](https://huggingface.co/kssrikar4/KidneySeg-UNet)
[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

Two-stage cascade pipeline for kidney and tumor segmentation in abdominal CT scans. Pre-trained models hosted on Hugging Face; inference via interactive Streamlit app.

## Architecture

| Stage | Purpose | Input | Output |
|-------|---------|-------|--------|
| **1 (Localization)** | Coarse kidney ROI extraction | 128³ downsampled dual-window CT | Binary kidney mask |
| **2 (Refinement)** | Fine-grained kidney + tumor segmentation | Native-res crop + Stage 1 mask (3 channels) | Multi-class mask (kidney=1, tumor=2) |

> Models: `SegResNet` (MONAI). Loss: `DiceFocalLoss` (γ=2.0). Pre-trained weights: [🤗 kssrikar4/KidneySeg-UNet](https://huggingface.co/kssrikar4/KidneySeg-UNet)

## Run Inference

### 1. Install dependencies
```bash
pip install torch monai streamlit simpleitk scipy huggingface_hub
```

### 2. Launch the Streamlit app
```bash
streamlit run app.py
```


### 3. Use the interface
- Upload DICOM series (folder) or NIfTI file (`.nii`/`.nii.gz`)
- Click **Run Segmentation**
- View side-by-side: original CT slice / fused segmentation overlay
- Toggle **Play Video** to scroll through volume
- Visualize kidney parenchyma (green) + tumor (red)
- Volume estimates displayed: Kidney / Tumor in mL *(voxel spacing: 1.5×1.5×3.0 mm)*

<img width="800" height="405" alt="tumor" src="https://github.com/user-attachments/assets/7cc847bd-5983-4674-a623-8ef9babdfc29" />

Example Patient – Only Tumor (Red)
<img width="800" height="405" alt="tumor+kidney" src="https://github.com/user-attachments/assets/9e956c59-bf2c-4bd5-a86b-9c4f939231f3" />

Example Patient – Kidney (Green) + Tumor (Red)

> Models auto-download from Hugging Face on first run. Requires internet access.

## Performance (Held-out Test Set)

| Stage | Metric | Score |
|-------|--------|-------|
| 1 (Localization) | Dice (kidney) | **0.87** |
| 2 (Refinement) | Dice (tumor) | **0.66** |

## Results on Test Set

<img width="1773" height="574" alt="eval_plot_KiTS-00209" src="https://github.com/user-attachments/assets/4c123195-06cd-4938-b694-af5d44bfe555" />
<img width="1773" height="574" alt="eval_plot_KiTS-00116" src="https://github.com/user-attachments/assets/7d6d68bb-1b03-41e8-b444-35859e1e92d1" />
<img width="1773" height="574" alt="eval_plot_KiTS-00102" src="https://github.com/user-attachments/assets/2ba4a9f7-506b-4a56-a6fb-0eafc8a7d6a7" />
<img width="1773" height="574" alt="eval_plot_KiTS-00071" src="https://github.com/user-attachments/assets/adb57bca-05c7-4add-be17-9ff5eca702ef" />
<img width="1773" height="574" alt="eval_plot_KiTS-00058" src="https://github.com/user-attachments/assets/78645ce0-c5cc-4c4f-9c08-347a989fff6f" />
<img width="1773" height="574" alt="eval_plot_KiTS-00057" src="https://github.com/user-attachments/assets/283d6fd6-fcb7-41f3-a02b-6ed54d8a95a7" />

*Note: Stage 2 operates on high-resolution, tumor-sparse crops; lower Dice reflects boundary-level difficulty, not failure.*

## Acknowledgements

- [KiTS](https://www.cancerimagingarchive.net/collection/c4kc-kits/) for dataset
>Please cite the original dataset when using this model:
>Heller, N., Sathianathen, N., Kalapara, A., Walczak, E., Moore, K., Kaluzniak, H., Rosenberg, J., Blake, P., Rengel, Z., Oestreich, M., Dean, J., Tradewell, M., Shah, A., Tejpaul, R., Edgerton, Z., Peterson, M., Raza, S., Regmi, S., Papanikolopoulos, N., & Weight, C. (2019). C4KC KiTS Challenge Kidney Tumor Segmentation Dataset (Version 3) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.2019.IX49E8NX
  
- [MONAI](https://monai.io/) for medical imaging tools
- [Hugging Face](https://huggingface.co/) for model hosting

# License

[Mozilla Public License Version 2.0](LICENSE) - Feel free to use and modify
