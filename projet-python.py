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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, f1_score
import xgboost
from xgboost import XGBClassifier
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
sm = SMOTE(k_neighbors=2, sampling_strategy=0.3)
Xsmoted, Ysmoted = sm.fit_resample(X_train6, Y8)

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
modelxgb = XGBClassifier(loss="deviance",learning_rate=0.3,max_depth=3,
                         max_features="sqrt",subsample=0.95,n_estimators=50)
modelxgb.fit(X6_train,Y6_train)
Y6_pred2 = modelxgb.predict(X6_test)
test_acc2 = accuracy_score(Y6_test, Y6_pred2)




#valeurs = lieux['Num_Acc']
#valeurs2 = valeurs.to_numpy()
#repet = usagers['Num_Acc']
#repet2 = repet.to_numpy()

#valeurs3 = vehicules['id_vehicule']
#valeurs4 = valeurs3.to_numpy()
#repet3 = usagers['id_vehicule']
#repet4 = repet3.to_numpy()

#but créer un df contenant les infos de test mais avec des lignes doublées en fn du nb de passagers dans la voiture

#def compte(valeur,repet):
    #compte le nombre de répétitions de lignes pour un accident dans test2 
    #L = []
    #nn = len(valeur)
    #for i in range(nn):
        #a = valeur[i]
        #c = np.count_nonzero(repet == a)
        #L.append(c)
   #return L
        

#liste = compte(valeurs2,repet2)
# comment passer de 56518 à 129153

#liste2 =compte(valeurs4,repet4)

#fruit_list = [ ('Orange', 34, 'Yes' )]
#df = pd.DataFrame(fruit_list, columns = ['Name' , 'Price', 'Stock'])

#def creat(vehicules,liste):
    #vehicules2 = vehicules.copy()
    #l = len(liste)
    #ll = 0
    #for i in range(l):
        #if liste[i] !=1:
            #for j in range(liste[i]-1):
                #vehicules2.loc[l+ll] = vehicules.iloc[i]
                #ll+=1
    #return vehicules2
            
#on crée les lignes doublons dans véhicules qui correspondent aux usagers d'un même véhicule afin de concat avec usagers ensuite par les colonnes
#vehicules2 = creat(vehicules,liste2)

#vehicules3 = vehicules2.sort_values(by = ['Num_Acc', 'id_vehicule'])    
    
#6 lignes de décalage dans les deux dataframes véhicules3 et usagers, jsp pq
#vehicules4 = vehicules3.reset_index(drop=True)
    

#bug = vehicules5['Num_Acc']
#bug2 = bug.to_numpy()
#bugprime = usagers['Num_Acc']
#bugprime2 = bugprime.to_numpy()   
   

#A = compte(valeurs2,bug2)
#B = compte(valeurs2,bugprime2)

#def diff(A,B):
#    l = len(A)
    #L = []
    #for i in range(l):
        #L.append(A[i]-B[i])
    #return L

#lignes de vehicules4 à drop : 
    #première de Num_Acc = 202100009430
    #dernière de Num_Acc = 202100022987
    #dernière de Num_Acc = 202100040202
    #première de Num_Acc = 202100040933
    #première de Num_Acc = 202100044328
    #première de Num_Acc = 202100053167
    
    
#vehicules5 = vehicules4.drop(index = [21787,52699,92651,94291,101765,121556])

#test2 = pd.concat([vehicules5,usagers], axis = 1)

