# TCP-Diffusion

This repository contains the testing code and a subset of the data for the TCP-Diffusion project.

⚠️ *To preserve anonymity for double-blind review, only the basic usage instructions are provided here. The full dataset and training code will be released upon acceptance.*

## 🔧 Environment Setup

Configure the necessary environment. Recommended versions:

- Python 3.7  
- PyTorch ≥ 1.10 with CUDA 10.2

---

## 📦 Download Model and Dataset Subset

Download the pre-trained model and the 2020 subset of the dataset via the following **anonymous link**:

**[Download Link](https://your-anonymous-link.com)**

⚠️ Due to hosting platform limitations, the link will expire on **YYYY-MM-DD** (one week from release). Please download it as soon as possible.

**Note:**
- Only data from the year **2020** is included.
- The full dataset will be released upon paper acceptance.

---

## 📁 Dataset Preparation

1. Extract the downloaded dataset to a path of your choice, e.g., `your_data_path`.
2. Edit **line 17** of the file: `video_diffusion_pytorch/rainfall_dataset_ICML.py`
3. Replace the default path with your actual dataset location.

---

## 📥 Model Placement

Place the pre-trained model `model-55.pt` into the following directory: `results/TCP_ICML_Test/`

---

## 🚀 Run Evaluation

To run the evaluation, use the following command:

```bash
python evaluation_new_F4noinit_E1_ifscat_0321_ICML.py ICMLtest \
    --save results/TCP_ICML_Test \
    --output_frames 4 \
    --train_batch_size 16 \
    --test_epoch 55 \
    --timesteps 200 \
    --new_split \
    --multi_modals tquvz_t2m_sst_msl_topo_ifs \
    --input_transform_key loge \
    --loss_type l2
```
## 🖼️ Output

A subset of visualization results will be saved to: `results/TCP_ICML_Test/ICML_under_review_55/predictions_55/`

