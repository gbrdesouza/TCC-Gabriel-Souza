#!/usr/bin/env python
# coding: utf-8

# In[14]:


pip install lightgbm


# In[12]:


pip install xgboost


# In[7]:


get_ipython().system('pip install numerize')
get_ipython().system('pip install catboost')


# In[1]:


import kaggle


# In[2]:


get_ipython().system('kaggle datasets list -s credit')


# In[3]:


get_ipython().system('kaggle datasets download -d nelgiriyewithana/credit-card-fraud-detection-dataset-2023')


# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections



from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline



from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv(r"C:\Users\ghabb\Downloads\credit card 2023\creditcard_2023.csv")


# In[3]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression                 # To scaled data with mean 0 and variance 1
from sklearn.model_selection import train_test_split                # To split the data in training and testing part
from sklearn.tree import DecisionTreeClassifier                     # To implement decision tree classifier
from sklearn.metrics import classification_report                   # To generate classification report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# In[4]:


print ('Visão do dataset -', data.shape)
data.head()


# In[5]:


#Vendo informações do dataset
data.info()


# In[6]:


#Descrição dos dados
data.describe()


# # Observações
# - Temos 568630 linhas de observações e 31 colunas
# - A variável 'Class' é o Output indicando se a transação foi fraudulenta (1) ou não (0)
# - Não há valores faltantes no banco de dados
# - Não há valores duplicados
# - O tipo de cada variável parece correto

# In[7]:


data.isnull().sum().max()


# In[9]:


data.columns


# In[10]:


data.isna().sum()


# In[11]:


data.duplicated().any()


# In[12]:


#Analisando o desbalanceamento entre as classes
print('Não fraudulentos:', round(data['Class'].value_counts()[0]/len(data)*100,2), '% do banco de dados')
print('Fraudulentos:', round(data['Class'].value_counts()[1]/len(data)*100,2), '% do banco de dados')


# In[13]:


data['Class'].value_counts()[1]


# In[14]:


data['Class'].value_counts()[0]


# In[17]:


colors = ["#0101DF", "#DF0101"]

sns.countplot(x='Class', data=data, palette = colors)
plt.title('Distribuição de "Class"')


# In[15]:


sns.kdeplot(data=data['Amount'], shade=True, color = "#6AA84F")
plt.show()

data['Amount'].plot.box()
#"Amount" se assemelha a uma distribuição normal


# In[16]:


azulvermelho = sns.diverging_palette(20, 220, as_cmap=True)
paper = plt.figure(figsize=[20,10])
sns.heatmap(data.corr(),cmap = azulvermelho, annot=True)
plt.show()


# # Matriz de correlações
# - V17 tem alta correlação com V16 e V18
# - V10 tem alta correlação com V9 e V12
# - V4 e V11 tem alta correlação negativa com V10, V12 e V14

# In[17]:


f, axes = plt.subplots(ncols=2, figsize=(15,4))
colors = ["#0101DF", "#DF0101"]

# Correlações negativas com Classe (quanto menor o valor, maior a probabilidade de ser fraude)
sns.boxplot(x="Class", y="V12", data=data, palette=colors, ax=axes[0])
axes[0].set_title('V12 vs "Class" - Correlação negativa')

sns.boxplot(x="Class", y="V14", data=data, palette=colors, ax=axes[1])
axes[1].set_title('V14 vs "Class" - Correlação negativa')




# In[18]:


f, axes = plt.subplots(ncols=2, figsize=(15,4))
colors = ["#0101DF", "#DF0101"]

# Correlações positivas com Classe (quanto maior o valor, maior a probabilidade de ser fraude)
sns.boxplot(x="Class", y="V4", data=data, palette=colors, ax=axes[0])
axes[0].set_title('V4 vs "Class" - Correlação positiva')


sns.boxplot(x="Class", y="V11", data=data, palette=colors, ax=axes[1])
axes[1].set_title('V11 vs "Class" - Correlação positiva')


# In[22]:


data.skew()


# In[19]:


from scipy.stats import norm

f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 6))

v1_fraud_dist = data['V1'].loc[data['Class'] == 1].values
sns.histplot(v1_fraud_dist,ax=ax1, color='#FB8861', stat="density", kde_kws=dict(cut=3), bins=50)
ax1.set_title('V1 Distribution \n (Fraud Transactions)', fontsize=14)

v10_fraud_dist = data['V10'].loc[data['Class'] == 1].values
sns.histplot(v10_fraud_dist,ax=ax2, color='#56F9BB', stat="density", kde_kws=dict(cut=3), bins=50)
ax2.set_title('V10 Distribution \n (Fraud Transactions)', fontsize=14)


v23_fraud_dist = data['V23'].loc[data['Class'] == 1].values
sns.histplot(v23_fraud_dist,ax=ax3, color='#C5B3F9', stat="density", kde_kws=dict(cut=3), bins=50)
ax3.set_title('V23 Distribution \n (Fraud Transactions)', fontsize=14)

plt.show()



# In[20]:


#Dividindo os dados em variáveis dependentes e independentes
x = data.drop(['id','Class'],axis=1)
y = data.Class


# In[21]:


x.head()


# In[22]:


print('x =',x.shape)
print('y =',y.shape)


# In[63]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate


# In[24]:


x_array = x.to_numpy()


# In[1]:


#Usando algoritmo de validação cruzada para regressão logística
model = LogisticRegression()
steps = list()
steps.append(('scaler', StandardScaler()))
steps.append(('model', model))
pipeline = Pipeline(steps=steps)


# In[41]:


cv = KFold(n_splits=10, random_state=None, shuffle=False)


# In[60]:


scoring_set = ['accuracy','precision','recall','roc_auc','f1']


# In[75]:


lr_scores = cross_validate(pipeline, x_array, y, cv=cv, scoring=scoring_set)

print('fit time: ',(np.mean(lr_scores['fit_time']))) 
print('score time: ',(np.mean(lr_scores['score_time'])))    
print('Acurácia média: ',(np.mean(lr_scores['test_accuracy']))) 
print('Precisão média: ',(np.mean(lr_scores['test_precision'])))    
print('Recall média: ',(np.mean(lr_scores['test_recall'])))     
print('AUC média: ',(np.mean(lr_scores['test_roc_auc'])))  
print('F1 média: ',(np.mean(lr_scores['test_f1']))) 

for score in lr_scores['test_roc_auc']:
    print("AUC desta etapa: ", score)


# In[27]:


#z-score
sc = StandardScaler()


# In[28]:


#aplicando z-score em x
x_scaled = sc.fit_transform(x) 


# In[29]:


x_scaled_df = pd.DataFrame(x_scaled,columns=x.columns)


# In[30]:


x_scaled_df.head()


# In[31]:


# Dividindo o dataset em treino e teste
x_treino,x_teste,y_treino,y_teste = train_test_split(x_scaled_df,y,test_size=0.25,random_state=15,stratify= y)


# In[32]:


print(x_treino.shape)
print(x_teste.shape)
print(y_treino.shape)
print(y_teste.shape)


# In[33]:


sum(y)/len(y)


# In[34]:


# Inicializando variável de acurácia
acuracia_treino = {}
acuracia_teste = {}


# In[35]:


# Inicializando variável de recall
recall_treino = {}
recall_teste = {}


# In[36]:


# Inicializando variável de F1 score
f1_treino = {}
f1_teste = {}


# In[37]:


# Inicializando variável de precisão
from sklearn.metrics import precision_score
precisao_treino = {}
precisao_teste = {}


# In[38]:


#Construindo um modelo de Regressão Logística
lr = LogisticRegression()
start = time.time()
lr.fit(x_treino,y_treino)
end = time.time()
tempo = end - start
print("Tempo de treinamento =",tempo)


# In[39]:


def aval_modelo(treino, teste):
  acuracia = accuracy_score(treino, teste)
  clas_rep = classification_report(treino, teste)
  print('Acurácia do modelo: ', round(acuracia, 4))
  print(clas_rep)


# In[40]:


preds_lr_treino = lr.predict(x_treino)
preds_lr_teste = lr.predict(x_teste)


# In[41]:


# Acurácia
acuracia_treino[lr] = accuracy_score(np.array(y_treino), preds_lr_treino)
acuracia_teste[lr] = accuracy_score(np.array(y_teste), preds_lr_teste)

# Recall
recall_treino[lr] = recall_score(np.array(y_treino), preds_lr_treino)
recall_teste[lr] = recall_score(np.array(y_teste), preds_lr_teste)

# F1 score
f1_treino[lr] = f1_score(np.array(y_treino), preds_lr_treino)
f1_teste[lr] = f1_score(np.array(y_teste), preds_lr_teste)

# Precisão
precisao_treino[lr] = precision_score(np.array(y_treino), preds_lr_treino)
precisao_teste[lr] = precision_score(np.array(y_teste), preds_lr_teste)


# In[42]:


print('Treino - Regressão Logística')
aval_modelo(y_treino,preds_lr_treino)


# In[43]:


print('Teste - Regressão Logística')
aval_modelo(y_teste, preds_lr_teste)


# In[44]:


lr_cf_treino = confusion_matrix(y_treino, preds_lr_treino, labels=[1,0])


# In[45]:


lr_cf_teste = confusion_matrix(y_teste, preds_lr_teste, labels=[1,0])


# In[46]:


## Plotando matriz de confusão
def graf_matriz_confusao (matriz_cf_treino,
                           matriz_cf_teste,
                           classes:list)->None:
    
    
# Calculando os percentuais
    somatorio_treino = np.sum(matriz_cf_treino, axis = 1)
    percentual_treino = matriz_cf_treino / somatorio_treino[:,np.newaxis]*100

    somatorio_teste = np.sum(matriz_cf_teste, axis = 1)
    percentual_teste = matriz_cf_teste / somatorio_teste[:,np.newaxis]*100

    rotulos_treino = [['{} \n({:.1f}%)'.format(val, porc) for val,porc in zip(row,porc_row)] for row, porc_row in zip(matriz_cf_treino, percentual_treino)]
    
    rotulos_teste = [['{} \n({:.1f}%)'.format(val, porc) for val,porc in zip(row,porc_row)] for row, porc_row in zip(matriz_cf_teste, percentual_teste)]

# Plotando    
    fig,axes = plt.subplots(1,2,figsize=(9,4))
    sns.heatmap(matriz_cf_treino,
                annot = np.array(rotulos_treino),
                fmt = '',
                cmap = 'Blues',
                cbar = False,
                square = True,
                linewidths = 0.7,
                linecolor = 'white',
                ax = axes[0])
    sns.heatmap(matriz_cf_teste,
                annot = np.array(rotulos_teste),
                fmt = '',
                cmap = 'Oranges',
                cbar = False,
                square = True,
                linewidths = 0.7,
                linecolor = 'white',
                ax = axes[1])
    
        # Adicionando VP, FN, FP, VN à matriz de treino
    axes[0].text(0.5, 0.65, 'VP', ha='center', va='center', fontsize=9, fontweight='bold')
    axes[0].text(1.5, 0.65, 'FN', ha='center', va='center', fontsize=8, fontweight='bold')
    axes[0].text(0.5, 1.65, 'FP', ha='center', va='center', fontsize=8, fontweight='bold')
    axes[0].text(1.5, 1.65, 'VN', ha='center', va='center', fontsize=8, fontweight='bold')
    axes[0].set_title('Matriz de Confusão de Treino',fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predição', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Classe', fontsize=10, fontweight='bold')
    axes[0].set_xticklabels(classes)
    axes[0].set_yticklabels(classes)
    axes[0].tick_params(rotation=0, size = 8)

    # Adicionando VP, FN, FP, VN à matriz de teste
    axes[1].text(0.5, 0.65, 'VP', ha='center', va='center', fontsize=9, fontweight='bold')
    axes[1].text(1.5, 0.65, 'FN', ha='center', va='center', fontsize=8, fontweight='bold')
    axes[1].text(0.5, 1.65, 'FP', ha='center', va='center', fontsize=8, fontweight='bold')
    axes[1].text(1.5, 1.65, 'VN', ha='center', va='center', fontsize=8, fontweight='bold')
    axes[1].set_title('Matriz de Confusão de Teste',fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predição', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Classe', fontsize=10, fontweight='bold')
    axes[1].set_xticklabels(classes)
    axes[1].set_yticklabels(classes)
    axes[1].tick_params(rotation=0, size = 8)
    

    fig.tight_layout()
    plt.show()


# In[49]:


graf_matriz_confusao(lr_cf_treino, lr_cf_teste, ["1","0"] )


# In[51]:


#Usando algoritmo de validação cruzada para árvores de decisão
steps2 = list()
steps2.append(('scaler', StandardScaler()))
steps2.append(('model', DecisionTreeClassifier()))
pipeline2 = Pipeline(steps=steps2)


# In[77]:


cv2 = KFold(n_splits=10, random_state=None, shuffle=False)


# In[78]:


dt_scores = cross_validate(pipeline2, x_array, y, cv=cv2, scoring=scoring_set)

print('fit time: ',(np.mean(dt_scores['fit_time']))) 
print('score time: ',(np.mean(dt_scores['score_time'])))    
print('Acurácia média: ',(np.mean(dt_scores['test_accuracy']))) 
print('Precisão média: ',(np.mean(dt_scores['test_precision'])))    
print('Recall média: ',(np.mean(dt_scores['test_recall'])))     
print('AUC média: ',(np.mean(dt_scores['test_roc_auc'])))  
print('F1 média: ',(np.mean(dt_scores['test_f1']))) 

for score in dt_scores['test_roc_auc']:
    print("AUC desta etapa: ", score)


# In[50]:


# Construindo modelo de Árvores de Decisão
start = time.time()
dt = DecisionTreeClassifier()
dt.fit(x_treino,y_treino)
end = time.time()
tempo = end - start
print("Tempo de treinamento =",tempo)


# In[50]:


preds_dt_treino = dt.predict(x_treino)
preds_dt_teste = dt.predict(x_teste)


# In[51]:


print('Treino - Árvore de Decisão')
aval_modelo(y_treino,preds_dt_treino)


# In[52]:


print('Teste - Árvore de Decisão')
aval_modelo(y_teste, preds_dt_teste)


# In[53]:


# Acurácia
acuracia_treino[dt] = accuracy_score(np.array(y_treino), preds_dt_treino)
acuracia_teste[dt] = accuracy_score(np.array(y_teste), preds_dt_teste)

# Recall
recall_treino[dt] = recall_score(np.array(y_treino), preds_dt_treino)
recall_teste[dt] = recall_score(np.array(y_teste), preds_dt_teste)

# F1 score
f1_treino[dt] = f1_score(np.array(y_treino), preds_dt_treino)
f1_teste[dt] = f1_score(np.array(y_teste), preds_dt_teste)

# Precisão
precisao_treino[dt] = precision_score(np.array(y_treino), preds_dt_treino)
precisao_teste[dt] = precision_score(np.array(y_teste), preds_dt_teste)


# In[54]:


dt_cf_treino = confusion_matrix(y_treino, preds_dt_treino, labels=[1,0])
dt_cf_teste = confusion_matrix(y_teste, preds_dt_teste, labels=[1,0])
graf_matriz_confusao(dt_cf_treino, dt_cf_teste, ["1","0"] )


# In[79]:


#Usando algoritmo de validação cruzada para florestas aleatórias
steps3 = list()
steps3.append(('scaler', StandardScaler()))
steps3.append(('model', RandomForestClassifier()))
pipeline3 = Pipeline(steps=steps3)


# In[86]:


cv3 = KFold(n_splits=5, random_state=None, shuffle=False)


# In[87]:


rf_scores = cross_validate(pipeline3, x_array, y, cv=cv3, scoring=scoring_set)

print('fit time: ',(np.mean(rf_scores['fit_time']))) 
print('score time: ',(np.mean(rf_scores['score_time'])))    
print('Acurácia média: ',(np.mean(rf_scores['test_accuracy']))) 
print('Precisão média: ',(np.mean(rf_scores['test_precision'])))    
print('Recall média: ',(np.mean(rf_scores['test_recall'])))     
print('AUC média: ',(np.mean(rf_scores['test_roc_auc'])))  
print('F1 média: ',(np.mean(rf_scores['test_f1']))) 

for score in rf_scores['test_roc_auc']:
    print("AUC desta etapa: ", score)


# In[55]:


# Construindo modelo de Floresta Aleatória

rf = RandomForestClassifier()
start = time.time()
rf.fit(x_treino, y_treino)
end = time.time()
tempo = end - start
print("Tempo de treinamento =",tempo)


# In[56]:


preds_rf_treino = rf.predict(x_treino)
preds_rf_teste = rf.predict(x_teste)


# In[57]:


print('Treino - Floresta Aleatória')
aval_modelo(y_treino,preds_rf_treino)


# In[58]:


print('Teste - Floresta Aleatória')
aval_modelo(y_teste, preds_rf_teste)


# In[59]:


# Acurácia
acuracia_treino[rf] = accuracy_score(np.array(y_treino), preds_rf_treino)
acuracia_teste[rf] = accuracy_score(np.array(y_teste), preds_rf_teste)

# Recall
recall_treino[rf] = recall_score(np.array(y_treino), preds_rf_treino)
recall_teste[rf] = recall_score(np.array(y_teste), preds_rf_teste)

# F1 score
f1_treino[rf] = f1_score(np.array(y_treino), preds_rf_treino)
f1_teste[rf] = f1_score(np.array(y_teste), preds_rf_teste)

# Precisão
precisao_treino[rf] = precision_score(np.array(y_treino), preds_rf_treino)
precisao_teste[rf] = precision_score(np.array(y_teste), preds_rf_teste)


# In[60]:


rf_cf_treino = confusion_matrix(y_treino, preds_rf_treino, labels=[1,0])
rf_cf_teste = confusion_matrix(y_teste, preds_rf_teste, labels=[1,0])
graf_matriz_confusao(rf_cf_treino, rf_cf_teste, ["1","0"] )


# In[80]:


#Usando algoritmo de validação cruzada para xgb
steps4 = list()
steps4.append(('scaler', StandardScaler()))
steps4.append(('model', XGBClassifier(objective = "binary:logistic")))
pipeline4 = Pipeline(steps=steps4)


# In[81]:


cv4 = KFold(n_splits=10, random_state=None, shuffle=False)


# In[82]:


xgb_scores = cross_validate(pipeline4, x_array, y, cv=cv4, scoring=scoring_set)

print('fit time: ',(np.mean(xgb_scores['fit_time']))) 
print('score time: ',(np.mean(xgb_scores['score_time'])))    
print('Acurácia média: ',(np.mean(xgb_scores['test_accuracy']))) 
print('Precisão média: ',(np.mean(xgb_scores['test_precision'])))    
print('Recall média: ',(np.mean(xgb_scores['test_recall'])))     
print('AUC média: ',(np.mean(xgb_scores['test_roc_auc'])))  
print('F1 média: ',(np.mean(xgb_scores['test_f1']))) 

for score in xgb_scores['test_roc_auc']:
    print("AUC desta etapa: ", score)


# In[51]:


# Construindo modelo de XGBClassifier

xgb = XGBClassifier(objective = "binary:logistic")
start = time.time()
xgb.fit(x_treino,y_treino)
end = time.time()
tempo = end - start
print("Tempo de treinamento =",tempo)


# In[52]:


preds_xgb_treino = xgb.predict(x_treino)
preds_xgb_teste = xgb.predict(x_teste)


# In[53]:


# Acurácia
acuracia_treino[xgb] = accuracy_score(np.array(y_treino), preds_xgb_treino)
acuracia_teste[xgb] = accuracy_score(np.array(y_teste), preds_xgb_teste)

# Recall
recall_treino[xgb] = recall_score(np.array(y_treino), preds_xgb_treino)
recall_teste[xgb] = recall_score(np.array(y_teste), preds_xgb_teste)

# F1 score
f1_treino[xgb] = f1_score(np.array(y_treino), preds_xgb_treino)
f1_teste[xgb] = f1_score(np.array(y_teste), preds_xgb_teste)

# Precisão
precisao_treino[xgb] = precision_score(np.array(y_treino), preds_xgb_treino)
precisao_teste[xgb] = precision_score(np.array(y_teste), preds_xgb_teste)


# In[54]:


print('Treino - XGBClassifier')
aval_modelo(y_treino,preds_xgb_treino)


# In[55]:


print('Teste - XGBClassifier')
aval_modelo(y_teste,preds_xgb_teste)


# In[56]:


xgb_cf_treino = confusion_matrix(y_treino, preds_xgb_treino, labels=[1,0])
xgb_cf_teste = confusion_matrix(y_teste, preds_xgb_teste, labels=[1,0])
graf_matriz_confusao(xgb_cf_treino, xgb_cf_teste, ["1","0"] )


# In[83]:


#Usando algoritmo de validação cruzada para lgbm
steps5 = list()
steps5.append(('scaler', StandardScaler()))
steps5.append(('model', LGBMClassifier ()))
pipeline5 = Pipeline(steps=steps5)


# In[84]:


cv5 = KFold(n_splits=10, random_state=None, shuffle=False)


# In[85]:


lgbm_scores = cross_validate(pipeline5, x_array, y, cv=cv5, scoring=scoring_set)

print('fit time: ',(np.mean(lgbm_scores['fit_time']))) 
print('score time: ',(np.mean(lgbm_scores['score_time'])))    
print('Acurácia média: ',(np.mean(lgbm_scores['test_accuracy']))) 
print('Precisão média: ',(np.mean(lgbm_scores['test_precision'])))    
print('Recall média: ',(np.mean(lgbm_scores['test_recall'])))     
print('AUC média: ',(np.mean(lgbm_scores['test_roc_auc'])))  
print('F1 média: ',(np.mean(lgbm_scores['test_f1']))) 

for score in lgbm_scores['test_roc_auc']:
    print("AUC desta etapa: ", score)


# In[74]:


# Construindo modelo de LGBMClassifier

lgbm = LGBMClassifier ()
start = time.time()
lgbm.fit(x_treino,y_treino)
end = time.time()
tempo = end - start
print("Tempo de treinamento =",tempo)


# In[75]:


preds_lgbm_treino = lgbm.predict(x_treino)
preds_lgbm_teste = lgbm.predict(x_teste)


# In[76]:


# Acurácia
acuracia_treino[lgbm] = accuracy_score(np.array(y_treino), preds_lgbm_treino)
acuracia_teste[lgbm] = accuracy_score(np.array(y_teste), preds_lgbm_teste)

# Recall
recall_treino[lgbm] = recall_score(np.array(y_treino), preds_lgbm_treino)
recall_teste[lgbm] = recall_score(np.array(y_teste), preds_lgbm_teste)

# F1 score
f1_treino[lgbm] = f1_score(np.array(y_treino), preds_lgbm_treino)
f1_teste[lgbm] = f1_score(np.array(y_teste), preds_lgbm_teste)

# Precisão
precisao_treino[lgbm] = precision_score(np.array(y_treino), preds_lgbm_treino)
precisao_teste[lgbm] = precision_score(np.array(y_teste), preds_lgbm_teste)


# In[77]:


print('Treino - LGBMClassifier')
aval_modelo(y_treino,preds_lgbm_treino)


# In[78]:


print('Teste - LGBMClassifier')
aval_modelo(y_teste,preds_lgbm_teste)


# In[79]:


lgbm_cf_treino = confusion_matrix(y_treino, preds_lgbm_treino, labels=[1,0])
lgbm_cf_teste = confusion_matrix(y_teste, preds_lgbm_teste, labels=[1,0])
graf_matriz_confusao(lgbm_cf_treino, lgbm_cf_teste, ["1","0"] )


# In[88]:


#Usando algoritmo de validação cruzada para catboost
steps6 = list()
steps6.append(('scaler', StandardScaler()))
steps6.append(('model', CatBoostClassifier()))
pipeline6 = Pipeline(steps=steps6)


# In[89]:


cv6 = KFold(n_splits=10, random_state=None, shuffle=False)


# In[90]:


cb_scores = cross_validate(pipeline6, x_array, y, cv=cv6, scoring=scoring_set)

print('fit time: ',(np.mean(cb_scores['fit_time']))) 
print('score time: ',(np.mean(cb_scores['score_time'])))    
print('Acurácia média: ',(np.mean(cb_scores['test_accuracy']))) 
print('Precisão média: ',(np.mean(cb_scores['test_precision'])))    
print('Recall média: ',(np.mean(cb_scores['test_recall'])))     
print('AUC média: ',(np.mean(cb_scores['test_roc_auc'])))  
print('F1 média: ',(np.mean(cb_scores['test_f1']))) 

for score in cb_scores['test_roc_auc']:
    print("AUC desta etapa: ", score)


# In[80]:


# Construindo modelo de CatBoostClassifier

cb = CatBoostClassifier() 
start = time.time()
cb.fit(x_treino,y_treino)
end = time.time()
tempo = end - start
print("Tempo de treinamento =",tempo)


# In[81]:


preds_cb_treino = cb.predict(x_treino)
preds_cb_teste = cb.predict(x_teste)


# In[82]:


# Acurácia
acuracia_treino[cb] = accuracy_score(np.array(y_treino), preds_cb_treino)
acuracia_teste[cb] = accuracy_score(np.array(y_teste), preds_cb_teste)

# Recall
recall_treino[cb] = recall_score(np.array(y_treino), preds_cb_treino)
recall_teste[cb] = recall_score(np.array(y_teste), preds_cb_teste)

# F1 score
f1_treino[cb] = f1_score(np.array(y_treino), preds_cb_treino)
f1_teste[cb] = f1_score(np.array(y_teste), preds_cb_teste)

# Precisão
precisao_treino[cb] = precision_score(np.array(y_treino), preds_cb_treino)
precisao_teste[cb] = precision_score(np.array(y_teste), preds_cb_teste)


# In[83]:


print('Treino - CatBoostClassifier')
aval_modelo(y_treino,preds_cb_treino)


# In[84]:


print('Teste - CatBoostClassifier')
aval_modelo(y_teste,preds_cb_teste)


# In[85]:


cb_cf_treino = confusion_matrix(y_treino, preds_cb_treino, labels=[1,0])
cb_cf_teste = confusion_matrix(y_teste, preds_cb_teste, labels=[1,0])
graf_matriz_confusao(cb_cf_treino, cb_cf_teste, ["1","0"] )


# In[68]:


#Plotando curva ROC de todos os modelos
lr_fpr, lr_tpr, lr_thresold = roc_curve(y_teste, preds_lr_teste)
dt_fpr, dt_tpr, dt_threshold = roc_curve(y_teste, preds_dt_teste)
rf_fpr, rf_tpr, rf_threshold = roc_curve(y_teste, preds_rf_teste)
xgb_fpr, xgb_tpr, xgb_threshold = roc_curve(y_teste, preds_xgb_teste)
lgbm_fpr, lgbm_tpr, lgbm_threshold = roc_curve(y_teste, preds_lgbm_teste)
cb_fpr, cb_tpr, cb_threshold = roc_curve(y_teste, preds_cb_teste)

def graph_roc_curve_multiple(lr_fpr, lr_tpr, dt_fpr, dt_tpr, rf_fpr, rf_tpr, xgb_fpr, xgb_tpr, lgbm_fpr, lgbm_tpr, cb_fpr, cb_tpr):
    plt.figure(figsize=(16,8))
    plt.title('Curva ROC', fontsize=30)
    plt.plot(lr_fpr, lr_tpr, label='Regressão Logística: {:.4f}'.format(roc_auc_score(y_teste, preds_lr_teste)))
    plt.plot(dt_fpr, dt_tpr, label='Árvores de Decisão: {:.4f}'.format(roc_auc_score(y_teste, preds_dt_teste)))
    plt.plot(rf_fpr, rf_tpr, label='Florestas Aleatórias: {:.4f}'.format(roc_auc_score(y_teste, preds_rf_teste)))
    plt.plot(xgb_fpr, xgb_tpr, label='XGBoost: {:.4f}'.format(roc_auc_score(y_teste, preds_xgb_teste)))
    plt.plot(lgbm_fpr, lgbm_tpr, label='LGBM: {:.4f}'.format(roc_auc_score(y_teste, preds_lgbm_teste)))
    plt.plot(cb_fpr, cb_tpr, label='CatBoost: {:.4f}'.format(roc_auc_score(y_teste, preds_cb_teste)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('Taxa de Falsos Positivos', fontsize=20)
    plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=20)
    plt.annotate('Valor mínimo: 50%', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()
    
graph_roc_curve_multiple(lr_fpr, lr_tpr, dt_fpr, dt_tpr, rf_fpr, rf_tpr, xgb_fpr, xgb_tpr, lgbm_fpr, lgbm_tpr, cb_fpr, cb_tpr)
plt.show()

