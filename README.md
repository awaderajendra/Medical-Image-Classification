## 🧠 Brain Cancer Detection using Deep Learning (CNN)

## 📝 Project Description
This project focuses on classifying MRI brain scans to detect brain cancer. 
Built using convolutional Neural networks (CNN), the model processes medical images to assist in medical 
diagnosis by identifying anomalies across 2 distinct classes.

## 📊 Dataset Details
**Total Images:** 6,056 MRI scans
**Classes:** 4 (Multi-class classification)
**Image Dimensions:** 128 into 128 pixels 

## 🧬 Model Architecture
* **Input Layer:** `(128, 128, 3)`
* **Convolutional Layer:** `Conv2D` with 32 filters, (3,3) kernel, and ReLU activation
* **Pooling Layer:** `MaxPooling2D` with (2,2) pool size
* **Flatten Layer:** Prepares features for the Dense layer
* **Dense Hidden Layer:** 128 units with ReLU activation
* **Output Layer:** 4 Units with Softmax activation (for categorical classification)

## ⚙️Technologies Used
* Python
* Tensorflow & Keras (Deep Learning)
* Flask (Web API)

# 📉 Results & Performance
The Convolutional Neural Network (CNN) achieved the following results:
Training Accuracy: 95%
Validation Accuracy: 92%

# 📁 Project Structure
```text
medical-image-classification/
|--Brain_Cancer raw MRI data/
   |--Brain_cancer/
|--.gitignore
|--app.py
|--train_model.py
|--README.md
|--requirements.txt
```
# 🚀 Future Enhancements
Train other advanced models like **VGG16** or **ResNet50** (Transfer learning) for better accuracy.
Build an interactive Web User Interface using **Streamlit** for instant MRI daignosis.

# 👩‍💻 Author
Madhuri Dhawade
