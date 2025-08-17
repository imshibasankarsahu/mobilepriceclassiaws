Mobile Price Classification with AWS SageMaker
Overview
This project demonstrates training and deploying a machine learning model for mobile price classification using AWS SageMaker with scikit-learn. A Random Forest Classifier predicts mobile phone price ranges (0-3) based on features like battery power, RAM, and others. The implementation uses a Jupyter Notebook with boto3 and SageMaker SDK, covering data preparation, S3 upload, model training, endpoint deployment, and inference.

Optionally, you can use Amazon SageMaker Canvas for a no-code alternative to build and deploy the model visually.

Dataset
Source: mob_price_classification_train.csv (place in project directory)

Features: battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory, m_dep, mobile_wt, n_cores, pc, px_height, px_width, ram, sc_h, sc_w, talk_time, three_g, touch_screen, wifi

Target: price_range (0: low, 1: medium, 2: high, 3: very high)

Shape: 2000 rows, 21 columns

Missing Values: None

Requirements
Python 3.8+

AWS account with SageMaker access

IAM role with SageMaker and S3 permissions (e.g., arn:aws:iam::788614365622:role/sagemakeraccess)

Dependencies
bash
pip install sagemaker boto3 pandas scikit-learn
Setup
AWS Credentials
Configure via ~/.aws/credentials or environment variables.

S3 Bucket
Uses mobsagemaker213. Replace with your bucket if needed.

Ensure bucket exists in your AWS region.

Dataset
Place mob_price_classification_train.csv in the working directory.

Usage
Jupyter Notebook Workflow
Open mobile_price_classification.ipynb (or equivalent)

Run cells in order:

Load libraries and initialize SageMaker session

Preprocess data and split into train/test sets

Save splits as CSV and upload to S3

Train model using SKLearn estimator

Deploy model to an endpoint

Perform predictions on test data

Delete endpoint to avoid costs

Key Code Snippets
Data Preparation
python
# Data preprocessing and train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Training Script (script.py)
Trains RandomForestClassifier, saves as model.joblib.

SageMaker Training
python
sklearn = SKLearn(
    entry_point='script.py',
    role=role,
    instance_type='ml.m5.large',
    framework_version='0.23-1'
)
sklearn.fit({'training': train_input})
Deployment
python
predictor = sklearn.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)
Prediction
python
print(predictor.predict(testX[features][:2].values.tolist()))
Cleanup
python
sm_boto3.delete_endpoint(EndpointName=endpoint_name)
SageMaker Canvas (No-Code Option)
Access SageMaker Canvas via AWS Console

Upload mob_price_classification_train.csv

Select Classification and set price_range as target

Train and deploy model using the Canvas UI

See SageMaker Canvas Documentation for more details.

AWS Resources
Training Job: RF-custom-sklearn-*

Endpoint: Custom-sklearn-model-*

S3 Paths: s3://mobsagemaker213/sagemaker/mobile_price_classification/sklearncontainer/*

Cost Note
Use spot instances for training and delete endpoints after use to minimize costs.

Metrics
Accuracy: ~0.88 (varies with data)

Output: Classification report with precision, recall, F1 per class

Troubleshooting
Permissions: Ensure IAM role has AmazonSageMakerFullAccess and S3 permissions

Instance Quotas: Verify availability for ml.m5.large/ml.m4.xlarge

S3 Issues: Check bucket name and region match

Model Artifact: Retrieved from S3 post-training

Improvements
Use SageMaker HyperparameterTuner for optimization

Explore deep learning (e.g., TensorFlow/PyTorch) for better accuracy

Implement CI/CD with AWS CodePipeline

License
MIT License. Free to use and modify.
