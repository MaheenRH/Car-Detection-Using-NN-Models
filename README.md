# 🚗 Car Detection Using Neural Network Models  

This project implements **Car Detection using Deep Learning and Computer Vision techniques**.  
The goal is to accurately identify and classify cars in images using Neural Network architectures built with Python and TensorFlow/Keras.

---

## 📘 Project Overview  

The **Car Detection Model** leverages convolutional neural networks (CNNs) to detect and localize vehicles from image data.  
It demonstrates the complete ML workflow — from data preprocessing and augmentation to model training, evaluation, and visualization.

The notebook `Car_Detection_Using_NN_Models.ipynb` includes:
- Image preprocessing and augmentation
- CNN model construction and training
- Evaluation on test images
- Visualization of predictions

---

## 🧠 Key Features  

- 🚘 **Image Classification / Object Detection** using Neural Networks  
- 🧩 **Data Augmentation** for robust training  
- ⚙️ **Custom CNN and transfer learning models** (optional extension)  
- 📈 **Training metrics visualization** (accuracy, loss curves)  
- 🧮 **Evaluation on unseen test images**  
- 💾 **Model saving and inference demo**

---

## 🧰 Tools & Technologies  

| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python 3.10 |
| **Deep Learning** | TensorFlow, Keras |
| **Data Handling** | NumPy, Pandas |
| **Image Processing** | OpenCV, Pillow |
| **Visualization** | Matplotlib, Seaborn |
| **Runtime** | Jupyter Notebook / Google Colab |

---

## ⚙️ Project Workflow  

1. **Data Preprocessing**
   - Load dataset of car images  
   - Resize and normalize pixel values  
   - Split into training and validation sets  

2. **Model Building**
   - Define CNN architecture using Keras Sequential API  
   - Use activation functions (ReLU, Softmax)  
   - Optimize with Adam optimizer  

3. **Training & Evaluation**
   - Train the model for multiple epochs  
   - Track accuracy and loss curves  
   - Evaluate performance on unseen images  

4. **Prediction Visualization**
   - Display detected cars and model confidence  
   - Annotate results using OpenCV or Matplotlib  

---

---


## 🖼 Example Results

Below are some of the car detections produced by the model:

<p align="center">
  <img src="https://github.com/user-attachments/assets/d625314b-9f6c-4bac-8716-1ab64a17ac18" width="45%" alt="Car Detection Example 1">
  <img src="https://github.com/user-attachments/assets/b7597929-3dec-4756-a558-473c4d62476c" width="45%" alt="Car Detection Example 2">
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/541d8fb2-0fbf-41d1-83be-d1e58a55e451" width="45%" alt="Car Detection Example 3">
</p>

<p align="center">
  <em>Bounding boxes with confidence scores (green) show detected vehicles in various lighting and perspectives.</em>
</p>

---

---

---

## 📊 Model Performance Comparison

The table below summarizes training and validation performance for different neural network architectures used in car detection.

| Model | Training Loss (MSE) | Training MAE | Training Precision | Validation Loss (MSE) | Validation MAE | Validation Precision |
|:------|---------------------:|--------------:|-------------------:|----------------------:|----------------:|---------------------:|
| **Model 1** | 2.6242 | 1.1814 | 0.9692 | 3.0446 | 1.2962 | 0.9799 |
| **Model 2** | 20381.8027 | 96.8078 | 0.9609 | 20647.9500 | 100.1070 | 0.9799 |
| **Model 3** | 16305.9500 | 83.9299 | 0.9692 | 18220.0020 | 84.3409 | 0.9799 |
| **Model 4** | 23666.9062 | 110.5713 | 0.9692 | 22488.1719 | 102.8278 | 0.9799 |

---

### 🧩 Insights
- **Model 1** shows the best performance overall with **lowest MSE** and **highest consistency** between training and validation metrics.  
- **Models 2–4** had higher losses but similar precision values, suggesting stable detection accuracy despite different architectures.  
- The optimized model achieves nearly **98% precision** on validation data, successfully detecting cars in varied lighting and perspective conditions.

---



