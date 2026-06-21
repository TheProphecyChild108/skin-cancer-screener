# Clinical Skin Cancer Classifier with Class-Imbalance Optimization

An advanced end-to-end computer vision pipeline built in PyTorch utilizing a pre-trained **ResNet50** backbone to classify dermatological lesions into 7 distinct diagnostic categories using the HAM10000 dataset.

## 🔬 Project Evolution & Engineering Highlights

### Phase 1: Identity Segregation (Anti-Data Leakage)
* **The Challenge:** Standard random validation splitting yielded a deceptive ~100% accuracy due to identical lesion variations appearing across the same patients in both splits.
* **The Solution:** Implemented a strict identity-level `GroupShuffleSplit` by `lesion_id`, ensuring complete isolation of patient backgrounds.

### Phase 2: Multi-Class Resolution
* **The Evolution:** Expanded the network from a rigid binary system to a 7-class diagnostic split. This resolved false-positive traps where benign, textured anomalies (e.g., Seborrheic Keratoses) were misidentified as generic malignancies.

### Phase 3: Class-Imbalance Optimization (Production Grade)
* **The Problem:** Extreme sample dominance of common moles skewed the model's prediction boundaries, leading to an initial Melanoma recall rate of just 37%.
* **The Mitigation:** 1. **Dynamic Cost-Sensitive Learning:** Computed inverse frequency class weights to penalize the `CrossEntropyLoss` function proportionally for rare class errors.
  2. **Stochastic Data Augmentation:** Applied random horizontal/vertical flips, affine rotations, and color jitter parameters to synthesize variations in sparse clinical vectors.

---

## 📊 Final Optimized Performance Metrics

Evaluated across a validation dataset of 989 completely unseen patient profiles, the optimized pipeline achieved an overall generalization accuracy of **80%**:

| Diagnostic Category | Precision | Recall (Sensitivity) | Support | Status vs. Unweighted |
| :--- | :---: | :---: | :---: | :--- |
| **Melanocytic nevi (nv)** | 0.98 | 0.98 | 404 | Solid Stability 🔒 |
| **Melanoma (mel)** | 0.87 | **0.57** | 215 | **Massive Recall Boost (+20%)** 🔥 |
| **Benign keratosis (bkl)** | 0.68 | 0.71 | 240 | Highly Consistent 🔍 |
| **Basal cell carcinoma (bcc)** | 0.72 | 0.80 | 88 | Clean Balance 📈 |
| **Actinic keratoses (akiec)** | 0.00 | 0.00 | 0 | Insufficient Validation Data |
| **Vascular lesions (vasc)** | 0.79 | 0.92 | 24 | Robust Generalization ✨ |
| **Dermatofibroma (df)** | 0.21 | **0.83** | 18 | **Explosive Sensitivity Lift** 🚀 |

---

## 🛠️ Repository Architecture

* `pipeline.py` - Manages group-based splits and integrates stochastic training-set augmentations.
* `model.py` - Configures the pre-trained ResNet50 deep residual neural backbone.
* `train.py` - Computes cost-sensitive inverse class weights and runs the optimization loops.
* `evaluate.py` - Generates multi-class tracking arrays and explicit classification matrices.
* `predict.py` - Live inference script for running diagnostics on independent external images.
* `app.py` - Live interactive Gradio web application layout.

---

## 🔒 Copyright & Terms of Use

Copyright © 2026. All rights reserved. 

This repository and its entire codebase are the exclusive intellectual property of the author. No part of this project—including the machine learning pipeline design, custom weighting matrices, or application code—may be copied, reproduced, redistributed, or repurposed in any format without explicit written permission from the creator.