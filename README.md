# Multiclass Skin Lesion Classification Using Cost-Sensitive ResNet50

This repository contains an end-to-end computer vision pipeline implemented in PyTorch to classify dermatological lesions into seven diagnostic categories using the HAM10000 dataset. The project focuses on mitigating severe class imbalance and preventing data leakage during validation.

## Methodology & Engineering Implementation

### 1. Data Partitioning and Leakage Prevention
* **Problem:** The HAM10000 dataset contains multiple images of identical lesions taken from the same patient across different timeframes or angles. A standard random train/validation split causes identical patient data to leak into both sets, resulting in artificially inflated validation accuracy.
* **Solution:** Implemented a patient-level split using `GroupShuffleSplit` grouped by `lesion_id`. This guarantees that no patient background present in the training set appears within the validation set, ensuring a true measurement of model generalization.

### 2. Transition from Binary to Multiclass Classification
The pipeline was expanded from a binary (benign/malignant) classifier to a seven-class system. This design choice prevents false positives caused by benign, highly textured lesions (such as Seborrheic Keratoses) being misclassified as generic malignancies, allowing for granular differential diagnostics.

### 3. Class Imbalance Mitigation
The dataset exhibits severe class imbalance, heavily weighted toward common melanocytic nevi. This skew caused early training iterations to default to the majority class, resulting in poor minority class sensitivity (Melanoma recall was initially 37%). Two methods were used to counter this:
* **Cost-Sensitive Learning:** Computed inverse-frequency weights applied directly to the `CrossEntropyLoss` function, heavily penalizing misclassifications on rare categories.
* **Data Augmentation:** Implemented stochastic transformations including random affine rotations, horizontal/vertical flips, and color jitter to expand the minority class representation during training batches.

---

## Evaluation Metrics

Evaluated across a validation partition of 989 patient-isolated profiles, the final model achieved an overall accuracy of 80%. 

| Diagnostic Category (Class ID) | Precision | Recall (Sensitivity) | Support |
| :--- | :---: | :---: | :---: |
| **Melanocytic nevi (nv)** | 0.98 | 0.98 | 404 |
| **Melanoma (mel)** | 0.87 | 0.57 | 215 |
| **Benign keratosis-like lesions (bkl)** | 0.68 | 0.71 | 240 |
| **Basal cell carcinoma (bcc)** | 0.72 | 0.80 | 88 |
| **Actinic keratoses (akiec)** | 0.00 | 0.00 | 0 |
| **Vascular lesions (vasc)** | 0.79 | 0.92 | 24 |
| **Dermatofibroma (df)** | 0.21 | 0.83 | 18 |

### Optimization Curve
![ResNet50 Optimization Progress Across Epochs](learning_curve.png)

### Diagnostic Confusion Matrix
![Lesion Classification Confusion Matrix](confusion_matrix.png)

*Note on performance:* The application of inverse-frequency weighting increased Melanoma recall to 57% (a net improvement of +20% over unweighted baselines) and significantly raised Dermatofibroma sensitivity to 83%, demonstrating a successful trade-off between absolute precision and clinical safety boundaries.

---

## File Architecture

* `pipeline.py` - Script handling dataset tokenization, group splits, and stochastic augmentations.
* `model.py` - Network architecture defining the pre-trained ResNet50 backbone.
* `train.py` - Training loop calculating class weights and executing optimization step functions.
* `evaluate.py` - Evaluation logic generating confusion matrices and classification reports.
* `predict.py` - Local inference pipeline for testing individual external images.
* `app.py` - Deployment script launching the interactive Gradio web application.

---

## Copyright & Terms of Use

Copyright © 2026. All rights reserved. 

This repository and its entire codebase are the exclusive intellectual property of the author. No part of this project—including the machine learning pipeline design, custom weighting matrices, or application code—may be copied, reproduced, redistributed, or repurposed in any format without explicit written permission from the creator.
