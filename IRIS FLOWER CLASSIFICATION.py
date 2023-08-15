#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


df = pd.read_csv("C:/Users/prade/Downloads/IRIS.csv")


# In[9]:


df.head()


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df['species'].value_counts()


# In[13]:


sns.countplot(data=df,x='species')


# In[14]:


sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species')


# In[15]:


sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species')


# In[16]:


sns.pairplot(data=df,hue='species');


# In[17]:


sns.heatmap(data=df.corr(),annot=True)


# In[18]:


X = df.drop('species',axis=1)
y = df['species']


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# In[21]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# In[22]:


log_model = LogisticRegression(solver='saga', multi_class = 'ovr', max_iter =5000)


# In[27]:


penalty = ['l1', 'l2', 'elasticnet']
l1_ratio = np.linspace(0,1,10)
C = np.logspace(0,10,10)

param_grid = {'penalty': penalty, 'l1_ratio': l1_ratio, 'C':C}
grid_model = GridSearchCV(log_model, param_grid=param_grid)
# To remove unwanted warnings in GridSearchCV
import warnings
warnings.filterwarnings('ignore') 


# In[28]:


grid_model.fit(scaled_X_train, y_train)


# In[25]:


grid_model.best_params_


# In[29]:


y_pred = grid_model.predict(scaled_X_test)
y_pred


# In[30]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
accuracy_score(y_test,y_pred)


# In[31]:


confusion_matrix(y_test,y_pred)


# In[32]:


print(classification_report(y_test,y_pred))


# In[33]:


ConfusionMatrixDisplay.from_estimator(grid_model,scaled_X_test,y_test)


# In[34]:


from sklearn.metrics import roc_curve, auc


# In[35]:


def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(5,5)):
    y_score = clf.decision_function(X_test)

    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # calculate dummies once
    y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
         # roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic example')
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
    ax.legend(loc="best")
    ax.grid(alpha=.4)
    sns.despine()
    plt.show()


# In[36]:


plot_multiclass_roc(grid_model, scaled_X_test, y_test, n_classes=3, figsize=(16, 10))


# In[ ]:




