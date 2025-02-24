# Heart Disease Classification  

This repository contains a machine learning pipeline for classifying heart disease using multiple models. The dataset used is `heart.csv`, which consists of medical attributes that help predict the presence of heart disease.  

## Dataset  
- The dataset includes various features such as age, cholesterol levels, blood pressure, and other cardiovascular-related measurements.  
- The `target` column represents the presence (`1`) or absence (`0`) of heart disease.
- link to dataset https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

## Models Used  
The notebook implements the following classification models:  

### Scale-Insensitive Models  
These models do not require feature scaling:  
- Random Forest (`RandomForestClassifier`)  
- Naive Bayes (`GaussianNB`)  
- Gradient Boosting (`GradientBoostingClassifier`)  

### Scale-Sensitive Models  
These models require standardized features:  
- k-Nearest Neighbors (`KNeighborsClassifier`)  
- Logistic Regression (`LogisticRegression`)  
- Support Vector Classifier (`SVC`)  

## Preprocessing and Training  
1. The dataset is split into training and testing sets using `train_test_split` (40% test size, random state set to 9).  
2. Feature scaling is applied to models that require it (`StandardScaler`).  
3. Each model is trained using the training data.  

## Model Evaluation  
- **Accuracy Scores**: Each model is evaluated on the test set.  
- **Recall Score**: Used to assess model performance, especially in detecting positive cases.  
- **ROC Curve & AUC Score**: Evaluated for `RandomForestClassifier`.  
- **Feature Importance**: Extracted and visualized for Random Forest.  

## Hyperparameter Tuning  
- `GridSearchCV` is used to optimize the hyperparameters of `RandomForestClassifier`.  
- The best model from the grid search is selected and evaluated.  

## Heatmap Analysis  
A heatmap is generated using `seaborn` to visualize feature correlations.  

## Viewing the Notebook  
To view the Jupyter Notebook online without downloading it, use the following nbviewer link:  
[View the Notebook](https://nbviewer.org/github/Rafal852/Heart-Disease-Detection/blob/main/heart_disease_prediction.ipynb))  

## Installation and Usage  

### Install Dependencies  
Ensure you have Python installed and run the following command:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
