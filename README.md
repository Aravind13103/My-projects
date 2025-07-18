# Pneumonia Detection Using Deep Learning 
## Project Overview

This project aims to develop a **Pneumonia Detection System** by leveraging **Transfer Learning** with pre-trained deep learning models such as **VGG16** and **EfficientNetB0**. It classifies **chest X-ray images** into *Normal* and *Pneumonia* categories. The solution also includes an **interactive GUI** built with **Tkinter**, allowing users to upload chest X-rays and receive automated predictions.

---

## Features
- Deep learning-based **chest X-ray classification**  
-  Transfer learning using **VGG16** and **EfficientNetB0**  
-  **Data augmentation** for better generalization  
-  **Tkinter GUI** for uploading images and making predictions  
-  Model performance visualization (loss/accuracy graphs)  
-  Efficient **training with mixed precision** for speed and performance  
-  Achieved **accuracy of ~98%** on training data and **~85%** on validation data

---

##  Tech Stack

| **Technology** | **Version** |
|----------------|-------------|
| Python         | 3.8 / 3.9   |
| TensorFlow     | 2.10+       |
| Keras          | 2.10+       |
| NumPy          | Latest      |
| Matplotlib     | Latest      |
| PIL (Pillow)   | Latest      |
| Tkinter        | Pre-installed with Python |

---

## Project Structure

```
├── chest_xray/                    # Dataset folder (train/test images)
├── our_model.h5                   # Trained model
├── train.py                       # Model training script
├── gui_app.py                     # Tkinter-based prediction interface
├── LossVal_loss.png               # Training/Validation loss graph
├── AccVal_acc.png                 # Training/Validation accuracy graph
└── README.md                      # Project documentation
```

---

## Installation

1. Clone the repository  
   ```
   git clone https://github.com/your-repo/pneumonia-detection.git
   cd pneumonia-detection
   ```

2. Install the required packages  
   ```
   pip install tensorflow numpy matplotlib pillow
   ```

---

## Dataset
- Dataset used: **Chest X-Ray Images (Pneumonia)** from Kaggle  
- Download link: [Kaggle Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- Folder structure:
  ```
  chest_xray/
      ├── train/
      │   ├── NORMAL/
      │   └── PNEUMONIA/
      └── test/
          ├── NORMAL/
          └── PNEUMONIA/
  ```

---

## How to Train the Model
1. **Training script:**  
   Run `train.py` to train the model:  
   ```
   python train.py
   ```

2. **Configuration:**  
   - Epochs: 5  
   - Batch Size: 16  
   - Model used: EfficientNetB0 (transfer learning)  
   - Learning Rate Scheduling: ReduceLROnPlateau  
   - Mixed precision enabled for faster training  

---

##  GUI Application (Tkinter)
1. Run the GUI app:
   ```
   python gui_app.py
   ```
2. Upload a chest X-ray image and click **Analyze**  
3. The app will display the uploaded image and predict whether the person is **Safe** or **Affected with Pneumonia**

---

