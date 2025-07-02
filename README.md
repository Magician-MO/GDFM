# GDFM: Generative Diffusion-Fused Model for Remote Sensing to Map Translation

***Chenxing Sun** & **Xixi Fan***
***GsimLab** from **China University of Geosciences***

This repository contains the official PyTorch implementation for the paper:  **Feature Constraints Map Generate Model Integrating Generative Adversarial and Diffusion Denoising** . Our framework introduces a novel hybrid model that synergizes Generative Adversarial Networks (GANs) and Diffusion Denoising Models (DMs) for high-fidelity map generation from remote sensing imagery.

![Figure 1](figures\fig1.png)

*Figure 1: The overall framework structure.*

## Key Features

* **Hybrid Generative Model**: Integrates a GAN backbone with a diffusion-style denoising module, ensuring both rapid generation and high-quality output.
* **Attention Mechanism**: A novel coordinate-channel attention mechanism is designed to enhance the discrimination of geographic features by modeling spatial and feature interactions.
* **High-Fidelity Output**: Achieves state-of-the-art performance in generating multi-scale tile maps with superior visual fidelity and improved evaluation scores (FID, SSIM, PSNR).
* **Robust Reconstruction**: Effectively reconstructs occluded roads and preserves complex topologies, overcoming common issues like blurring and discontinuity found in traditional models.

## Results

Our model demonstrates significant improvements over existing methods in qualitative comparisons. It excels at preserving road continuity, generating natural curves, and reconstructing details in complex scenes.

#### Qualitative Comparison

![Figure 2](figures\fig2.png)

*Figure 2: Comparison with other common image translation models (Pix2Pix, CycleGAN, Pix2PixHD).*

#### Detailed View

![Figure 3](figures\fig3.png)

*Figure 3: Our model shows superior performance in reconstructing occluded and fine roads.*

## Setup

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/Magician-MO/GDFM.git
cd GDFM
pip install -r requirements.txt
```

## Usage

### Datasets

Download the Maps or MLMG datasets and place them in the `./datasets/` directory. Your directory structure should look like this:

```
GDFM
├── datasets
│   ├── MAPS
│   │   ├── train
│   │   ├── val
│   │   └── test
├── ...
└── train.py
```

### Training

To train the model, run the following command. Adjust the parameters like `--name` , `--dataroot` , and `--gpu_ids` as needed.

```bash
python train.py --name PROJECT --model gdfm --batch_size 48 --direction AtoB --dataroot ./datasets/DATASET --gpu_ids 0,1,2,3
```

### Testing

To test a trained model, use the following command:

```bash
python test.py --name PROJECT --model gdfm --direction AtoB --dataroot ./datasets/DATASET
```

## Acknowledgments

This work is built upon the foundational concepts of several incredible projects. We extend our gratitude to the creators of:

* **Pix2Pix** : For providing the baseline GAN framework for image-to-image translation.
* **DDM UNet** : For the insights into diffusion denoising probabilistic models.
* **ATME** : For the inspiration drawn from their work on GAN discriminators.
