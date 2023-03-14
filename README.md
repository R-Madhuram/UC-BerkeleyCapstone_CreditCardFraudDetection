# Credit Card Fraud Detection
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/competitions/ieee-fraud-detection/data) [![Language](https://img.shields.io/badge/Lang-Python-brightgreen)](https://www.python.org/) [![ML Library](https://img.shields.io/badge/ML-Scikit--learn-yellowgreen)](https://scikit-learn.org/) [![AutoML Library](https://img.shields.io/badge/AutoML-pycaret-blue)](https://pycaret.gitbook.io/docs/) [![Capstone Project](https://img.shields.io/badge/CapstoneProject-UCBerkeley-yellow)](https://em-executive.berkeley.edu/professional-certificate-machine-learning-artificial-intelligence?utm_source=bing&utm_medium=c&utm_term=berkeley%20machine%20learning%20certification&utm_location=86946&utm_campaign=B-365D_US_BG_SE_BH-PCMLAI_Brand&utm_content=Berkeley_MLAI&msclkid=84b4646fc6111be42dcd007fcfb213cf)

## Project Description:

**Motivation:**

The number of digital transactions in financial sector is rapidly growing especially after the Covid-19 imposed restriction, which caused an upsurge in online shopping. Financial institutions hence have to come up with novel and efficient approaches to detect fraudulent transactions to ensure monetary safety and reliability for the customer and to prevent losses occurring to the institution itself due to such unscrupulous transactions.

**Business Objective:**

Building architectures and processes to help find the fraudulent transaction(s) and to distinguish them from regular transactions is of immense value to Businesses. This can be achieved through new technologies like Artificial Intelligence to address the growing digital presence of monetary transactions. 

**How AI and Machine Learning can Help?**

Using Machine learning and Artificial Intelligence we can find patterns that describe and distinguish the fraudulent and non-fraudulent transactions based on the various details associated with each transaction. This could be achieved using both labeled (data that has been fraudulent or non-fraudulent) or unlabeled data as ML techniques are capable of reading patterns that human eye can miss. In this project we selected a dataset provided by IEEE in Kaggle that contains labeled data and built classification models using this data. We propose to further improve this model by augmenting it with unsupervised techniques (anomaly detection). 

**Challenges we faced:**

The dataset was complex with high dimensions and observations of which many of the features were obfuscated for privacy purposes. This hindered us from using domain knowledge to perform dimensionality reduction. This also presented with us with unique challenges in tuning more complex ML algorithms as the training became computationally expensive. Also due to the inherent nature of the business question the dataset was highly imbalanced (97% vs 3% of class representation) with many non-fraud transactions compared to fraudulent transactions. This deterred the performance of classification models. 

## Methodology:

**A. Setting the data for business objective :**
1. EDA:
	- Systematic reduction of features using correlation analysis
	- Merging the training dataset features to make one single data ready for modeling. 

**B. Building the baseline and simple classification model(s):**
1.  Baseline:
	- Since the majority class is non-fraudulent data (class label - 0), we set the predictions based on this class .
2. Simple classification model(s):
	- Decision tree and Logistic regression was used to build basic classification model(s) as other ML models like KNN and SVM were inappropriate to handle dataset of this size. 

**C. Ensemble Model(s):**
1.  Ensemble techniques: 
	- We used Boosting classifiers like cat boost, adaboost, extreme gradient boosting, light gradient boosting (Hist Boosting in sk-learn) and stacked them to improve their efficiency. 
2. SMOTE sampling:
	- The ensemble and staked classification models were used along SMOTE sampling technique to improve their performance.

**D. AUTO-ML:**
1. Ensemble techniques:
	- Due to the size of the dataset and large number of tunable parameters, Pycaret library, which is a Auto-ML learning library
	   was utilized to further tune the models built in step C. 
2. SMOTE Sampling:
	- Auto-ML is efficient to address any imbalances present in the dataset. However, the option to fix the imbalance using SMOTE sampling was also utilized to explore the possibility of improving the ensemble model built with Auto-ML performance. 

**ML Data Pipeline:**

![ML Data Pipeline](https://user-images.githubusercontent.com/115378526/224903775-908099eb-8d0d-49ff-a266-061c07104f56.jpg)

## Executive Summary:

Predicting transactions that are fraudulent are of utmost importance not only to banking systems but also to almost every other business today with he rise of online business. Further even credit card transactions are based on electronic systems which are prone to hacking and hence online theft of transaction identity and monetary loss. 

Identification of such transactions can be hugely improved using artificial intelligence as they are capable of sifting through enormous amount of data that consist of information regarding the normal and fraudulent transactions. These are based on mathematical modeling that can detect  patterns which describe and distinguish the two classes of transactions. 

We were presented with a dataset that had ~500K transactions, of which ~3.5% represented fraudulent transactions. Models were built with increasing level of complexity catering the needs of the dataset which were 1. Many features (250 parameters) 2. Big data (~ 500K observations) 3. Obfuscated features (real names of the features are masked for privacy). 

We reduced the number of features systematically using correlation analysis wherein one feature was retained for every correlated pair of features. After reducing the dimensions we built baseline model that would just provide the classification based on the best guess which would be to predict the majority of the two classes (hence always predict non-fraud). 

Model with increasing complexity were built to predict the transactions better and a ML technique (BOOSTING methods) that combines several weak models to build a strong one presented with the highest classification score (~0.93 ROC-AUC score). Three such powerful boosting methods (cat-boost-0.89, XgBoost-0.93, lightGBM-0.92 ROC-AUC score) were stacked to build a meta model that presented with a classification score of 0.94. This meta classifier has already been stored in a cloud platform. 

Future work could be helpful to combine unsupervised learning techniques like anomaly detection (for which models have been built and presented) that could be placed on top the meta classifier in cloud to improve the fraudulent transaction identification.  

**Technology Stack Used:**

![Technology Stack](https://user-images.githubusercontent.com/115378526/224903725-8fadbb25-81ad-49d2-ae37-ea3d35dded11.jpg)


## Technical Summary:

In the first set of deliverables of this study we explored and proposed various data mining (DM) and Machine Learning/Artificial Intelligence (ML/AI) approaches to detect such fraudulent transactions by distinguishing them from the regular digital withdrawals/payments. We analyzed a huge dataset with approximately 500K observations and ~435 features. The features were split into two different datasets counting the transactions and identity information exclusively. A stepwise approach of Exploratory data analysis (EDA) and feature engineering was employed that allowed to select a reduced set of 157 features. We compared two ML models namely Logistic regression and Decision tree along with boosting methods for their success in identifying fraudulent transactions successfully. The models were evaluated based on their area under the receiver operating characteristic curve (ROC-AUC) and recall values. Due to the inherent presence of class imbalance (about 96% of data is normal transaction), we compared the AUC metric of the models with a baseline model followed by recall to improve the reliability of the model we built. The models we built were tested on three evaluation sets using the stratified k-fold method.

Based on the initial model building exercise we found that Logistic regression with balanced was able to present the highest ROC-AUC score of ~0.84. It performed better than the base model in producing high recall score of ~0.75. However the precision of this model was poor (0.10) leading to the decrease in accuracy. 

In the next iteration, we performed ensemble machine learning to build meta models to improve the performance of the individual models. Specifically, we tested, boosting techniques (adaboost, light GBM, cat boost, extreme gradient boosting (XG Boosting)) and stacked the boosted models using various meta classifiers. We also utilized SMOTE (Synthetic Minority Oversampling Technique) to address the imbalance of the dataset. In the first iteration we performed SMOTE sampling on the both training and test dataset. 

Although the stacked classifier with Random forest meta classifier and lightGBM(HistGradient Boosting), cat boost and XG Boosting stacked classifiers was able to distinguish the fraud and non-fraud transaction with ~0.94 ROC-AUC score, we disregarded the model as we found that performing SMOTE on the test dataset led to model overfitting. We corrected the issue by sampling only the training dataset and then stacking the classifiers. However, tuning the parameters of the meta models was computationally expensive and time consuming.

To address this issue we chose Auto-ML techniques using Pycaret Python library which was 1. supported by GPU for all the stacked classifier models, 2. Automatically tuned the hyper parameters 3. Performed stratified k-fold with 10 folds 4. Fixed imbalance in the dataset. 

Ensemble model built using the Auto-ML tool gave superior performance when compared to models built without this tool. For instance CatBoost, lightGBM and XgBoost presented themselves with 0.89, 0.92 and 0.94 ROC-AUC score. This score was obtained with taking into account the inherent imbalance present in the dataset which when considered might improve the overall performance of the model(s).

These three ensemble model were then stacked along with various other models (decision tree, adaboost, light GBM and random forest) to improve their individual performance. We found that Adaboost and light GBM performed equally well with a ROC-AUC score of 0.94 and a recall score of 0.53. Hence we chose LightGBM (or AdaBoost) as the final model to be stored in the Azure cloud platform . 

This model was further tested by fixing the class imbalance using SMOTE sampling technique provided but the Pycaret library. We found that it decreased the ROC-AUC score to ~0.87. This could be attributed to the fact that SMOTE sampling might actually be making the model learn the synthetic data patterns leading to poor performance on the actual dataset. 

## ROADMAP (guide to navigate through project and notebooks):

* data:
	* ieee-fraud-detection.zip
* helperfunctions:
	* helper_functions.py (file operations helper function)
	* helper_functions_ml.py (helper functions for analysis & model evaluation)
* notebooks:
	* 1_eda:
		* 1_Dataloading_and_train_transaction_EDA.ipynb (load the transaction data and EDA on it).
		* 2_train_identity_EDA.ipynb (EDA on the training identity data)
		* 3_Feature_Engineering.ipynb (feature engineering on the transaction and identity data)
	* 2_simpleclassificationmodels:
		* 4_Baseline_model.ipynb (split data in to K folds and baseline model analysis)
		* 5_Logistic_Regression.ipynb (train and comparisons between different logistic regression models)
		* 6_Decision_Trees.ipynb (train and comparisons between different decision trees)
	* 3_ensemble_methods:
		* 7_Ensemble_methods.ipynb (Hist Gradient Classifier - original data and sampled data)
		* 8_CatBoost.ipynb (CAT Boosting Classifier - original data and sampled data)
		* 9_XGBoost.ipynb (Extreme Gradient Boosting Classifier - original data and sampled data)
	* 4_AutoML_ensembleModels:
		* 13_Auto_ML_Model.ipynb (Auto ML models - ensemble and stacked)
		* 14_Auto_ML_Model.ipynb (Auto ML models - ensemble and stacked on stacked data)
* Classification_Products:
	* columns_to_retain.csv - features to retain from training transaction data
	* columns_to_retain_identity.csv - features to retain from training identity data
	* df.csv - feature engineered data
	* train_dev_indices.pickle - training and dev row indices for each fold in the 3 fold cross validation
	* model_metrics_hyperparamtune.csv - logistic regression model metrics after hyper parameter tuning
	* model_metrics_nohyperparamtune.csv - logistic regression model metrics after no hyper parameter tuning

**Notebook 1 and 2 (Data loading and EDA)**: We loaded the datasets (transaction and identity dataset). Built box and bar charts for the numerical and categorical variables (NOTE: since we conduct this analysis to get a high level view of the dataset, the axis might not be on scale and would require zooming in). The respective notebook flows to get deeper into various grouped features, categorizes them on basis of their null structures and reduces features based on correlation. The reduced features are then joined and saved in the respective notebook. The results of each set of analysis is summarized at the end of each sub-heading. 

**Notebook 3 (Feature Engineering)**: We load the reduced feature from the previous two notebooks and merge them to construct the  complete dataframe. In this dataframe we scale chosen numerical columns, identify and reduce the groups within categorical columns with very varied data (like deice type etc), drop unnecessary column(s), label encode the categorical columns and construct new features by combining existing features. We then save the feature engineered dataframe that is ready for building models. 

**Notebook 4, 5, and 6 (ML Models)**: We now build a baseline model (Notebook 4) to set the yardstick for the other sophisticated models. We started with a simple logistic regression model and increase it complexity (balance weights, grid search, Adaboost) to predict the fraudulent transactions (Notebook 5). We then proceed to another type of model Decision tree model and fine-tune to find the model performance for the prediction of fraudulent data (Notebook 6)

(NOTE: This project is in alignment with the submission of problem statement and methods for the capstone project in Module 17. It is to be noted that the mentioned submission (problem statement of this project) is different from the capstone project idea proposed in Module 11 submission (initial question of capstone and data). The change in capstone topic was motivated by lack of ease in fetching the dataset for the Module 11 capstone idea, which was pertaining to analysis of electric vehicles.)
