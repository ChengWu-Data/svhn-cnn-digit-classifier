# SVHN CNN Digit Classifier

This project implements a convolutional neural network for classifying cropped digit images from the Street View House Numbers (SVHN) dataset.

The goal is to train a CNN on the SVHN training set and evaluate it on the official SVHN test set. The model is designed for the Format 2 cropped digit files:

- `train_32x32.mat`
- `test_32x32.mat`

The dataset is not included in this repository because the assignment asks us not to submit the data files.

## Project goal

The task is a 10-class image classification problem. Each image is a 32 by 32 RGB image containing one digit from a real-world house number photo. The model predicts one of the digit classes from 0 to 9.

SVHN labels use the value `10` for digit `0`, so the code converts label `10` into label `0` before training.

## Repository contents

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── cw3729_ChengWu_individualProject.py
