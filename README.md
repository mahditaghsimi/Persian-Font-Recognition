# Persian Font Recognition using Deep Learning

This repository implements an end-to-end **Persian font recognition system** based on deep learning, covering the entire pipeline from **synthetic dataset generation** to **training, evaluation, and analysis**.

The project focuses on robust font classification by combining **HTML-based data generation**, **advanced image augmentation**, and a **multi-channel CNN architecture enhanced with attention mechanisms and handcrafted structural features**.

---

## Key Features

- **Synthetic Dataset Generation**
  - HTML + Playwright–based rendering
  - Real Persian fonts (TTF/OTF)
  - RTL and Persian text support
  - Text and digit image generation

- **Advanced Data Augmentation**
  - Geometric transformations (rotation, affine, resize)
  - Photometric effects (brightness, contrast, blur, noise)
  - Background color replacement & split backgrounds
  - Text highlighting and color combinations
  - Fully configurable via `.env` file

- **Deep Learning Model**
  - 4-channel input representation:
    - Original grayscale image
    - Edge features (Canny + Sobel)
    - Corner features (Harris)
    - Gradient magnitude
  - Multi-scale convolutional blocks
  - Channel Attention & Spatial Attention
  - Global average & max pooling
  - Label smoothing and mixed-precision training

- **Comprehensive Evaluation**
  - Confusion matrix with percentages
  - Per-class accuracy, precision, recall, and F1-score
  - ROC curves and AUC for all fonts
  - Top-K accuracy analysis
  - Prediction confidence distribution
  - Detailed error analysis and visual reports

---


