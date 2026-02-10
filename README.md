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

##  Installation & Dependencies
All required libraries for this project are listed in the `requirements.txt` file.  
To install the dependencies, follow the steps below.
---
#### 1️⃣ Prerequisites
- Python **3.8 or higher**
- An up-to-date version of `pip`

It is strongly recommended to create and activate a virtual environment before installation:
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```
---

#### 2️⃣ Install dependencies from requirements.txt
Run the following command in the root directory of the project:

```bash
pip install -r requirements.txt
```

- This command will automatically install all required libraries for:
  - Synthetic dataset generation
  - Data augmentation
  - Model training
  - Model evaluation and visualization
---

#### 3️⃣ Install Playwright browser dependencies
To run the make_data.py script (HTML-based dataset generation), Playwright requires an additional browser installation:

```bash
playwright install
```

⚠️ Without running this command, HTML-based dataset generation will not work.

---
#### 4️⃣ Important note on Persian text rendering
For correct rendering of Persian text (RTL layout and proper character shaping), the following libraries are included in ``requirements.txt``:

    - arabic-reshaper
    - python-bidi
If all dependencies are installed correctly, Persian text will be rendered properly in the generated images.

---

## 🚀 Usage

#### 1️⃣ Generate Dataset
```bash
python make_data.py
```
---
#### 2️⃣ Apply Data Augmentation
```bash
python augmentations.py
```
---
#### 3️⃣ Train the Model
```bash
python model.py
```
The trained model and label encoder will be saved automatically.

---
#### 4️⃣ Evaluate the Model
```bash
python evaluate.py
```
All evaluation plots and reports will be saved to the output directory.

---
##  Output Examples
- Confusion Matrix (with percentages)
- Per-class performance metrics
- ROC curves (per font)
- Top-K accuracy charts
- Sample predictions with confidence
- Error distribution analysis
---
## Applications
- Persian document analysis
- OCR preprocessing
- Font classification and verification
- Research on script-specific deep learning models
---
#### 📌 Notes
- The dataset is synthetically generated and does not rely on publicly available labeled font datasets.
- All augmentation and rendering parameters are fully configurable.
- The system is designed for extensibility to other scripts or languages.

---

### License
This project is intended for research and educational purposes.

---