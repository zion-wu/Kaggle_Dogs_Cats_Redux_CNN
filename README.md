# Dogs vs. Cats Classification with CNNs

This project applies convolutional neural networks (CNNs) to classify images of dogs and cats, based on the [Kaggle Dogs vs. Cats Redux: Kernel Edition](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition) competition. The goal is to compare multiple CNN architectures and evaluate how increasing model complexity and applying transfer learning impacts classification performance.

## 🧠 Management/Research Question

**In layman’s terms:**  
How can we teach machines to distinguish between dogs and cats from photographs with high accuracy, and why would this be useful?

This technology has real-world impact:
- In **e-commerce** (e.g., pet stores), accurate classification improves search and recommendation systems.
- In **veterinary care**, image recognition can help quickly identify pet types and even suggest breed-related risks.
- In **adoption platforms**, auto-tagging animals helps streamline listings.
- In **security systems**, distinguishing pets from humans can reduce false alarms.
- The same techniques apply to **wildlife monitoring**, **medical imaging**, and **smart agriculture**.

(LeCun, Bengio, and Hinton 2015; Simonyan & Zisserman 2015)

## 📊 Dataset Overview

- **Training Set**: 25,000 labeled images (12,500 dogs + 12,500 cats)
- **Test Set**: 12,500 unlabeled images for Kaggle evaluation
- All images resized to 150x150 pixels for CNN input

## 🧰 Preprocessing & Data Augmentation

- Resized all images to 150×150 pixels
- Scaled pixel values to [0,1]
- Performed 80/20 split for train/validation
- Applied horizontal flipping as augmentation
- Used `ImageDataGenerator` and `flow_from_dataframe` for efficient loading

## 🧪 Model Architectures

### 1️⃣ Simple CNN (Baseline)
- 2 convolutional layers + max pooling
- Dropout: 0.6
- Optimizer: Adam (lr=0.0001)
- Training accuracy: **86.9%**, validation: **79.3%**
- Overfitting observed

### 2️⃣ Deeper CNN
- More convolutional layers + BatchNorm + Dropout (0.5)
- Optimizer: Adam (lr=0.0001)
- Training accuracy: **87.8%**, validation: **84.8%**
- Improved generalization over baseline

### 3️⃣ VGG16 (Transfer Learning)
- Pre-trained VGG16 with fine-tuning
- Optimizer: Adam (lr=0.0001)
- Training accuracy: **96.4%**, validation: **92.2%**
- Best performance overall

All models trained for 10 epochs with batch size = 32

## 📈 Evaluation Metrics

- **Accuracy & Loss Curves**: Training vs. Validation across epochs
- **Confusion Matrices**: Identify misclassifications
- **ROC Curves & AUC**: Evaluate binary classification performance

| Model            | Validation Accuracy | Kaggle Score |
|------------------|---------------------|--------------|
| Simple CNN       | 79.3%               | —            |
| Deeper CNN       | 84.8%               | 0.43822      |
| VGG16 (Transfer) | 92.2%               | 0.24155      |


## 🧪 Methodology Highlights

- Cross-validation not used per epoch due to image size, but validation split ensured generalization.
- Hyperparameters (learning rate, dropout) tuned manually.
- Chose Adam optimizer for stability.
- Performance improved notably with pre-trained CNN (VGG16).
- Training time recorded for each model using `datetime` module.
