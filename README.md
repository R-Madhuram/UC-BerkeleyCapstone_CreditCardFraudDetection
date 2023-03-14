# Credit Card Fraud Detection
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

A. Setting the data for business objective :
1. EDA:
	i. Systematic reduction of features using correlation analysis
	ii. Merging the training dataset features to make one single data ready for modeling. 

B. Building the baseline and simple classification model(s):
1.  Baseline:
	i. Since the majority class is non-fraudulent data (class label - 0), we set the predictions based on this class .
2. Simple classification model(s):
	i. Decision tree and Logistic regression was used to build basic classification model(s) as other ML models like KNN and SVM were inappropriate to 
  handle dataset of this size. 

C. Ensemble Model(s):
1.  Ensemble techniques: 
	i. We used Boosting classifiers like cat boost, adaboost, extreme gradient boosting, light gradient boosting (Hist Boosting in sk-learn) and stacked them to improve their efficiency. 
2. SMOTE sampling:
	i. The ensemble and staked classification models were used along SMOTE sampling technique to improve their performance.

D. AUTO-ML:
1. Ensemble techniques:
	i. Due to the size of the dataset and large number of tunable parameters, Pycaret library, which is a Auto-ML learning library
	   was utilized to further tune the models built in step C. 
2. SMOTE Sampling:
	i. Auto-ML is efficient to address any imbalances present in the dataset. However, the option to fix the imbalance using SMOTE sampling was also utilized to explore the possibility of improving the ensemble model built with Auto-ML performance. 

## ROADMAP (guide to navigate through project and notebooks):

•	This project is split into multiple notebooks that have been numbered (along with titles) in accordance with the order of study, operation, and execution. Within each notebook the headings and subheading sections (when opened in collab appears in the side bar) provide the birds-eye view of the contents. 

•	Each heading/subheading of the first two notebook is followed by summary section that compiles the major findings of that section. The charts and visualizations that were utilised to just get a high level view of the data requires bigger/screen or zooming. All other visualizations have proper scaling of axis. 

•	Each notebook pertaining to the model offers a summary section at the end that compiles the results of the various models of that notebook and discusses the finding, points of roadblocks and future directions along with subheading of different version of the models and evaluation of the different versions. 

•	The notebooks are also supported by two separate python files that contain many helper functions to make repetitive taks like model evaluation easier. 

**Notebook 1 and 2 (Data loading and EDA)**: We loaded the datasets (transaction and identity dataset). Built box and bar charts for the numerical and categorical variables (NOTE: since we conduct this analysis to get a high level view of the dataset, the axis might not be on scale and would require zooming in). The respective notebook flows to get deeper into various grouped features, categorizes them on basis of their null structures and reduces features based on correlation. The reduced features are then joined and saved in the respective notebook. The results of each set of analysis is summarized at the end of each sub-heading. 

**Notebook 3 (Feature Engineering)**: We load the reduced feature from the previous two notebooks and merge them to construct the  complete dataframe. In this dataframe we scale chosen numerical columns, identify and reduce the groups within categorical columns with very varied data (like deice type etc), drop unnecessary column(s), label encode the categorical columns and construct new features by combining existing features. We then save the feature engineered dataframe that is ready for building models. 

**Notebook 4, 5, and 6 (ML Models)**: We now build a baseline model (Notebook 4) to set the yardstick for the other sophisticated models. We started with a simple logistic regression model and increase it complexity (balance weights, grid search, Adaboost) to predict the fraudulent transactions (Notebook 5). We then proceed to another type of model Decision tree model and fine-tune to find the model performance for the prediction of fraudulent data (Notebook 6)

(NOTE: This project is in alignment with the submission of problem statement and methods for the capstone project in Module 17. It is to be noted that the mentioned submission (problem statement of this project) is different from the capstone project idea proposed in Module 11 submission (initial question of capstone and data). The change in capstone topic was motivated by lack of ease in fetching the dataset for the Module 11 capstone idea, which was pertaining to analysis of electric vehicles.)
