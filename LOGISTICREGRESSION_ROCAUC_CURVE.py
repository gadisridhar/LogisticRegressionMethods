#!/usr/bin/env python
# coding: utf-8

# In[2]:



############################ For regression: f_regression, mutual_info_regression
############################ For classification: chi2, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, f_classif, mutual_info_classif, mutual_info_regression
from time import time
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, accuracy_score, auc, classification_report, f1_score, plot_roc_curve, roc_auc_score, roc_curve


# In[3]:


df = pd.read_csv('cancer.csv')
df


# In[4]:


sns.heatmap(df.corr())


# In[5]:


df['diagnosis'].replace(['M','B'],[1,0], inplace=True)
df


# In[6]:


df['diagnosis'].value_counts()


# In[7]:


Y = df.iloc[:, 31].values
print(Y)


# In[8]:


X = df.iloc[:, :31].values
print(X)


# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


# In[10]:


print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
print(type(X_train))
print(type(Y_train))


# In[11]:


scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)
print(X_train.shape)
print(X_test.shape)


# In[12]:


lr = LogisticRegression()
lr.fit(X_train, Y_train)


# In[13]:


lr.coef_


# # Y = B0 + B1X1 + B2X2+............+B31X31
# # no of columns 31
# # coefficients 31
# # Beta0 = 0.21340652,  Beta1 = 0.42474766,  Beta2 = 0.41003318,  0.3857563 ,  0.46057766,
# #  0.01591622, -0.5274524 ,  0.79462345,  1.13652457, -0.23377215,
# # -0.05797617,  1.28804102, -0.16761233,  0.62210769,  0.90554228,
# #  0.21231162, -0.69390044, -0.19099693,  0.31154038, -0.46438409,
# # -0.65072567,  0.86505766,  1.32315846,  0.57266121,  0.83329411,
# #  0.60115414, -0.00463997,  0.95613535,  0.78612981,  1.18444572,
# #  Beta31 = 0.17182771
# 

# In[14]:


Y_Pred_train = lr.predict(X_train)
Y_Pred_test = lr.predict(X_test)


# In[15]:


print(Y_Pred_train.shape)
print(Y_Pred_test.shape)


# In[16]:


acc_Score_train =  accuracy_score(Y_train,Y_Pred_train)
acc_Score_test = accuracy_score(Y_test,Y_Pred_test)
print("TRAIN SCORE : ", acc_Score_train)
print("TEST SCORE : ", acc_Score_test)


# In[17]:


cmTrain  = confusion_matrix(Y_train, Y_Pred_train)
print(cmTrain)


# In[18]:


cmTest =  confusion_matrix(Y_test, Y_Pred_test)
cmTest


# In[19]:


classifyTestReport = classification_report(Y_test, Y_Pred_test)
print(classifyTestReport)


# In[20]:


classifyTrainReport = classification_report(Y_train, Y_Pred_train)
print(classifyTrainReport)


# In[21]:


# TEST DATA GETTING TPR AND FPR WITH ROC_CURVE FUNCTION
Y_PredProba_test = lr.predict_proba(X_test)


# In[22]:


fprTest,tprTest,thresholdTest = roc_curve(Y_test, Y_PredProba_test[:,1], pos_label=1)
print(fprTest)
print(tprTest)
print(thresholdTest)


# In[23]:


# TRAIN DATA GETTING TPR AND FPR WITH ROC_CURVE FUNCTION
Y_PredProba_train = lr.predict_proba(X_train)


# In[24]:


fprTrain,tprTrain,thresholdTrain = roc_curve(Y_train, Y_PredProba_train[:,1], pos_label=1)


# In[25]:


print(fprTrain)
print(tprTrain)
print(thresholdTrain)


# In[26]:


#plot roc curves
plt.plot(fprTest, tprTest, linestyle='--',color='red', label='Logistic Regression')
plt.plot(fprTrain, tprTrain, linestyle='--',color='green', label='Logistic Regression')
#plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('TEST DATA - TRAIN DATA - ROC curve')
# x label
plt.xlabel('False Positive Rate - (1-Specificity)')
# y label
plt.ylabel('True Positive rate - (Sensitivity)')
plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




