import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import svm
import sklearn.metrics
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE,SMOTENC
from imblearn.over_sampling import RandomOverSampler

from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, f1_score
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
import os


os.chdir('/Users/paul-antoine/Desktop/projet-python-2A/accident-route')

lieux = pd.read_csv('lieux-2021.csv',sep= ";")
usagers = pd.read_csv('usagers-2021.csv',sep= ";")
vehicules = pd.read_csv('vehicules-2021.csv', sep= ";")
caracteristiques = pd.read_csv("carcteristiques-2021.csv",sep = ";")


lieux.isnull().sum()
usagers.isnull().sum()
vehicules.isnull().sum()
caracteristiques.isnull().sum()

test = pd.merge(caracteristiques, lieux)
test2 = pd.merge(usagers,vehicules, how= 'left')
test3 = pd.merge(test,test2, how= 'right')

test3.isnull().sum()
coltodrop = ['v2','lartpc','occutc'] #les NaN out
test3.drop(coltodrop,axis = 1, inplace = True)
mycol = ['lum','dep','com','agg','int','atm','col','lat','long','catr','circ','nbv',
         'prof','plan','surf','infra','situ','vma','catv','obs','obsm','choc','manv',
         'place','catu','secu1','secu2','secu3','grav']
test4 = test3[mycol]

## Préprocessing
#calculer le nombre de -1 dans chaque colonne = non renseigné

def renseignement(test4):
    L = []
    n,p = test4.shape
    for j in range(p):
        c = 0
        for i in range(n):
            if test4.iloc[i,j] == -1:
                c+=1
        L.append(c/n)
    return L
            
L = renseignement(test4)

def indices(L):
    indices = []
    nn = len(L)
    for i in range(nn):
        e = L[i]
        if e>0.2:
            indices.append(i)
    return indices
            
indices = indices(L)            
#indices des colonnes de test4 avec plus de 20% de valeurs non renseignées
#correspond à sécu2 et sécu3
#soit on vire ces colonnes, soit on met les -1 (non renseignés) à  0 (aucun équipement)

indemne = len(test4[test4.grav == 1])
blésséléger = len(test4[test4.grav == 4])
blésséhospi = len(test4[test4.grav == 3])
tué = len(test4[test4.grav == 2])

indexNames = test4[ test4['grav'] == -1 ].index
test4.drop(indexNames , inplace=True)

gravité = [indemne, blésséléger, blésséhospi, tué]


hist = plt.hist(test4['grav'], bins = [1,2,3,4,5],color = 'yellow',
            edgecolor = 'red')
plt.xlabel('gravité : 1 = indemne, 2 = tué, 3 & 4 = hospitalisé')
plt.ylabel('nombres')
plt.title('distribution gravité accidents')

labels, counts = np.unique(test4['grav'], return_counts=True)
plt.bar(labels, counts, align='center')
plt.gca().set_xticks(labels)
plt.show()

#déséquilibré : à prendre en compte dans l'algo dans le training set 
#: on placera des weight

sns.countplot(x="an_nais",data=test3)
plt.yscale("linear")
plt.title("années de naissance",fontsize=20)
plt.show()

mycol2 = ['lum','int','atm'
          ,'situ','vma','catv','obsm','choc','manv',
         'place','secu1','grav']
test6 = test4[mycol2]

test6.var()
on retire plan,surf et circ = très peu de variance 

features = ['lum','int','atm'
          ,'situ','catv','obsm','choc','manv',
         'place','secu1']

#on encode les variables catégoriques
X6 = pd.get_dummies(test6[features].astype(str))
#on met la gravité à 0 pour la première classe
test6['grav'] =  test6['grav'].apply(lambda x: x-1)
Y6 = test6['grav'].copy()
X_train6 = normalize(X6.values)

#on regroupe blessé hospitalisé et tué en catégorie 1
Y7 = test6['grav'].apply(lambda x: 1 if x==2 else x).copy()
Y8 = Y7.apply(lambda x: 2 if x == 3 else x)

labels2, counts2 = np.unique(Y8, return_counts=True)
plt.bar(labels2, counts2/len(Y8)*100, align='center')
plt.gca().set_xticks(labels2)
plt.show()

#XGBOOST = chiant pour les hyperparamètres
#réseaux de neurones ?
#SVM ?

X,y = test5.iloc[:,:-1], test5.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
model = svm.SVC(C=1,class_weight ="balanced")
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

sns.heatmap(confusion_matrix(y_test, y_pred), annot = True)

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=model.classes_)
disp.plot()
plt.show()

test5 = test4.copy()
test5['grav'] =  test5['grav'].apply(lambda x: 1 if x == 2 else 0)

X5 = test5.iloc[:,:-1]
y5 = test5.iloc[:,-1]
scaler = StandardScaler()
X5 = scaler.fit_transform(X5)
X_train5, X_test5, y_train5, y_test5 = train_test_split(X5, y5, test_size=0.20, random_state=42)
model5 = svm.SVC(C=1,class_weight ="balanced")
model5.fit(X_train5,y_train5)
y_pred5 = model5.predict(X_test5)
accuracy5 = sklearn.metrics.accuracy_score(y_test5, y_pred5)

cm5 = confusion_matrix(y_test5, y_pred5,labels=model5.classes_)
disp5 = ConfusionMatrixDisplay(cm5,display_labels=model5.classes_)
disp5.plot()
plt.show()

plt.figure(figsize=(12,10))
cor = test4.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


def compte(test4):
    L = []
    n,p = test4.shape
    for j in range(p):
        c = 0
        for i in range(n):
            if test4.iloc[i,j]==-1:
                c+=1
        L.append(c/n)
    return L
#on regarde les colonnes avec bcp de données non renseignées (-1)
                
Y6 = test6['grav']
X_train6 = normalize(X6.values)

#REBALANCE
sm = SMOTE()
Xsmoted, Ysmoted = sm.fit_resample(X_train6, Y8)
np.unique(Ysmoted, return_counts = True)

labelssm, countsm = np.unique(Ysmoted, return_counts=True)
plt.bar(labelssm, countsm/len(Ysmoted)*100, align='center')
plt.gca().set_xticks(labelssm)
plt.show()

#BEFORE REBALANCE

#RANDOM FOREST
xtrain, xtest, ytrain, ytest = train_test_split(X_train6,Y8)
model_rf = RandomForestClassifier(n_estimators=100, 
                                  max_depth=8)
model_rf.fit(xtrain, ytrain)
ypredrf = model_rf.predict(xtest)
test_acc = accuracy_score(ytest, ypredrf)

cmrf = confusion_matrix(ytest, ypredrf,labels=model_rf.classes_)
disprf = ConfusionMatrixDisplay(cmrf,display_labels=model_rf.classes_)
disprf.plot()
plt.show()

recall = recall_score(ytest, ypredrf, average='weighted')
f1 = f1_score(ytest, ypredrf, average='weighted')

#XGBOOST
#modelxgb = XGBClassifier(objective='multi:softprob',learning_rate=0.3,max_depth=3,
                         #max_features="sqrt",subsample=0.95,n_estimators=50)
#modelxgb.fit(X6_train,Y6_train)
#Y6_pred2 = modelxgb.predict(X6_test)
#test_acc2 = accuracy_score(Y6_test, Y6_pred2)

#AFTER REBALANCE
xtrain_reb, xtest_reb, ytrain_reb, ytest_reb = train_test_split(Xsmoted,Ysmoted)

#RANDOM FOREST
model_rf_prime = RandomForestClassifier(n_estimators=50, 
                                  max_depth=5)
model_rf_prime.fit(xtrain_reb, ytrain_reb)
ypredrf_reb = model_rf_prime.predict(xtest_reb)
test_acc_reb = accuracy_score(ytest_reb, ypredrf_reb)

cmrf_reb = confusion_matrix(ytest_reb, ypredrf_reb,labels=model_rf_prime.classes_)
disprf_reb = ConfusionMatrixDisplay(cmrf_reb,display_labels=model_rf_prime.classes_)
disprf_reb.plot()
plt.title('Confusion matrix with Random Forests')
plt.show()

recall_reb = recall_score(ytest_reb, ypredrf_reb,average ='macro')
f1_reb = f1_score(ytest_reb, ypredrf_reb,average = 'macro')

#premier esssai RF rebalanced voir cmrf_reb, 58% de précision sur 3 classes égalitaires.

#XGBOOST
modelxgb_reb = XGBClassifier(objective='multi:softprob',learning_rate=0.3,max_depth=3,subsample=0.95,n_estimators=30)
modelxgb_reb.fit(xtrain_reb,ytrain_reb)

ypredxgb_reb = modelxgb_reb.predict(xtest_reb)
test_acc_reb_xgb = accuracy_score(ytest_reb, ypredxgb_reb)
#63% de précision 
cmxgb_reb = confusion_matrix(ytest_reb, ypredxgb_reb,labels=modelxgb_reb.classes_)
dispxgb_reb = ConfusionMatrixDisplay(cmxgb_reb,display_labels=modelxgb_reb.classes_)
dispxgb_reb.plot()
plt.title('Confusion matrix with XGBOOST')
plt.show()

recall_xgbreb = recall_score(ytest_reb, ypredxgb_reb,average ='macro')
f1_xgbreb = f1_score(ytest_reb, ypredxgb_reb,average = 'macro')

#GRID SEARCH XGBOOST
#dico des paramètres qu'on veut tester
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5],
        'learning_rate' : [0.1,0.3,0.5,0.7],
        'n_estimators': [25,50,75,100,200]
        }

xgb = XGBClassifier( objective='multi:softprob',silent=True, nthread=1)
#5-cross val avec 10 essais de random_search
folds = 5
param_comb = 10

skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state =42)

random_search = RandomizedSearchCV(xgb, param_distributions=params, 
                                   n_iter=param_comb,scoring='accuracy', n_jobs=-1,
                                   cv=skf.split(xtrain_reb,ytrain_reb), verbose=3, random_state=42)

random_search.fit(xtrain_reb,ytrain_reb)

print(random_search.cv_results_)
print(random_search.best_estimator_)
print(random_search.best_params_)

#{'subsample': 0.6, 'n_estimators': 200, 'min_child_weight': 10, 'max_depth': 5, 'learning_rate': 0.5, 'gamma': 1.5}
#CV1 : 0.682, CV2 :0.680, CV3 : 0.679, CV4 : 0.680, CV5 : 0.676

xgb_best = XGBClassifier(objective='multi:softprob',learning_rate=0.5,
                         max_depth=5,subsample=0.6,n_estimators=200,
                         min_child_weight = 10, gamma = "1.5")
xgb_best.fit(xtrain_reb,ytrain_reb)

ypredxgb_best = xgb_best.predict(xtest_reb)
test_acc_reb_xgb_best = accuracy_score(ytest_reb, ypredxgb_best)
accu_xgb_best = round(test_acc_reb_xgb_best,3)*100

cmxgb_best = confusion_matrix(ytest_reb, ypredxgb_best,labels=xgb_best.classes_)
dispxgb_best = ConfusionMatrixDisplay(cmxgb_best,display_labels=xgb_best.classes_)
dispxgb_best.plot()
plt.title('Confusion matrix with XGBOOST after randomized search tuning : accuracy ='+str(accu_xgb_best)+'%')
plt.show()