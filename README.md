# 📱 Mobile Price Classification with AWS SageMaker

## 🚀 Overview
A machine learning project using **AWS SageMaker** and **scikit-learn** to classify mobile phones into 4 price ranges (0–3).  
- Model: **Random Forest Classifier**  
- Workflow: Data prep → S3 upload → Train → Deploy → Predict  

## 📂 Dataset
- **File**: `mob_price_classification_train.csv`  
- **Rows/Cols**: 2000 × 21  
- **Target**: `price_range` (0–3)  
- **Features**: battery_power, ram, px_height, px_width, etc.  

## ⚙️ Requirements
- Python 3.8+, AWS SageMaker + S3 access  
```bash
pip install sagemaker boto3 pandas scikit-learn
```

## 📒 Usage
1. Preprocess + split dataset  
2. Upload to S3  
3. Train with **SKLearn Estimator**  
4. Deploy endpoint  
5. Predict & cleanup  

## 📊 Results
- Accuracy: ~0.88  
- Classification report (precision/recall/F1 per class)  

## 🚀 Improvements
- Hyperparameter tuning  
- Try deep learning models  
- CI/CD with CodePipeline  

## 📜 License
MIT License – free to use & modify.
