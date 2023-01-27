# AI that knows when it does not know

## FLARE22 codebase of team brananas

This repository contains the codebase of team brananas (Natalia Alves & Bram de Wilde)
for the [FLARE22](https://flare22.grand-challenge.org/) MICCAI challenge.
This challenge features 50 labelled and 2000 unlabelled abdominal CT
scans, on which 13 organs have to be annotated.
We propose an uncertainty-guided self-learning framework to attack this
semi-supervised organ segmentation problem.
You can read our paper [here](https://doi.org/10.1007/978-3-031-23911-3_11).

## Training

All models are trained with the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
framework. Make sure to first install all dependencies in `requirements.txt`.

Training follows an iterative process:

1. Train 5 models (5-fold cross validation) with (pseudo-)labelled data
2. Perform inference on unlabelled data
3. Calculate uncertainty between the 5 models for each case
4. Include pseudolabelled cases that pass the uncertainty threshold
5. Repeat from step 1

Note that you have to manually edit some paths in files for this to
work. Please refer to the nnU-Net documentation for training and inference
of the models.

### 1. Train 5 models in 5-fold cross validation

```
nnUNet_train 3d_fullres Task101_FLARE [0, 1, 2, 3, 4]
```

### 2. Perform inference on unlabelled data

```
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t Task101_FLARE -m 3d_fullres
```

### 3. Calculate uncertainty metrics

```
python uncertainty_metrics/calculate_metrics.py
```

### 4. Include pseudolabelled cases that pass the uncertainty threshold

```
python ensemble.py
python generate_iteration.py
```

## Inference

To speed up nnU-Net inference, we built a custom inference pipeline
with some slight changes.
The final inference command also includes custom postprocessing.

```
python docker_inference_final.py
```

## Docker

You can build a Dockerfile for this model as follows:

```
docker build . --tag brananas:latest
```

## Citation

If you use this framework, or parts of it, please cite us as follows:

```
@InProceedings{10.1007/978-3-031-23911-3_11,
author="Alves, Nat{\'a}lia
and de Wilde, Bram",
editor="Ma, Jun
and Wang, Bo",
title="Uncertainty-Guided Self-learning Framework for Semi-supervised Multi-organ Segmentation",
booktitle="Fast and Low-Resource Semi-supervised Abdominal Organ Segmentation",
year="2022",
publisher="Springer Nature Switzerland",
address="Cham",
pages="116--127",
isbn="978-3-031-23911-3"
}
```
