# UC-BerkeleyCapstone_CreditCardFraudDetection
## ABSTRACT

The number of digital transactions in financial sector is rapidly growing especially after the Covid-19 imposed restriction, which caused an upsurge in online shopping. Financial institutions hence have to come up with novel and efficient approaches to detect fraudulent transactions to ensure monetary safety and reliability for the customer and to prevent losses occurring to the institution itself due to such unscrupulous transactions. 

In this study we explore and propose various data mining (DM) and Machine Learning/Artificial Intelligence (ML/AI) approaches to detect such fraudulent transactions by distinguishing them from the regular digital withdrawals/payments. We analyzed a huge dataset with approximately 500K observations and ~435 features. The features were split into two different datasets counting the transactions and identity information exclusively. A stepwise approach of Exploratory data analysis (EDA) and feature engineering was employed that allowed to select a reduced set of 157 features. We compared two ML models namely Logistic regression and Decision tree along with boosting methods for their success in identifying fraudulent transactions successfully. The models were evaluated based on their area under the receiver operating characteristic curve (ROC-AUC) and recall values. Due to the inherent presence of class imbalance (about 96% of data is normal transaction), we compared the accuracy metric of the models with a baseline model to improve the reliability of the model we built. The models we built we tested on three evaluation sets using the stratified k-fold  method. 

Based on the initial model building exercise we found that Logistic regression with balanced has been able to present the highest ROC-AUC score of ~0.84. It performs better than the base model in producing high recall score of ~0.75. However the precision of this model is poor (0.10) leading to the decrease in accuracy. 

Hence we propose to utilize 
1. boosting methods to improve the sophistication of the model to be able to distinguish the classes more accurately which in turn can boos the precision score and ROC-AUC score. The boosting methods, AdaBoost, CatBoost, XGBosst and LightGBM (HistClassifier in SK learn) are some attractive options that would be explored in the next phase of this project. Also stacking these ensemble methods might improve their individual scores. 
2. Deep learning methods like LSTM-GRU RNN and Graph models that could also be employed to detect the fraudulent transactions of this dataset and compared with ML models. 

## ROADMAP (guide to navigate through project and notebooks):

•	This project is split into multiple notebooks that have been numbered (along with titles) in accordance with the order of study, operation, and execution. Within each notebook the headings and subheading sections (when opened in collab appears in the side bar) provide the birds-eye view of the contents. 

•	Each heading of the first two notebook is followed by summary section that compiles the major findings of that section. 

•	Each notebook pertaining to the model offers a summary section at the end that compiles the results of the various models of that notebook and discusses the finding, points of roadblocks and future directions along with subheading of different version of the models and evaluation of the different versions. 

Notebook 1 and 2 (Data loading and EDA): We loaded the datasets (transaction and identity dataset). Built box and bar charts for the numerical and categorical variables (NOTE: since we conduct this analysis to get a high level view of the dataset, the axis might not be on scale and would require zooming in). The respective notebook flows to get deeper into various grouped features, categorizes them on basis of their null structures and reduces features based on correlation. The reduced features are then joined and saved in the respective notebook. The results of each set of analysis is summarized at the end of each sub-heading. 

Notebook 3 (Feature Engineering): We load the reduced feature from the previous two notebooks and merge them to construct the  complete dataframe. In this dataframe we scale chosen numerical columns, identify and reduce the groups within categorical columns with very varied data (like deice type etc), drop unnecessary column(s), label encode the categorical columns and construct new features by combining existing features. We then save the feature engineered dataframe that is ready for building models. 

Notebook 4, 5, and 6 (ML Models): We now build a baseline model (Notebook 4) to set the yardstick for the other sophisticated models. We started with a simple logistic regression model and increase it complexity (balance weights, grid search, Adaboost) to predict the fraudulent transactions (Notebook 5). We then proceed to another type of model Decision tree model and fine-tune to find the model performance for the prediction of fraudulent data (Notebook 6)
