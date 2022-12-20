import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#os.chdir('/Users/paul-antoine/Desktop/projet python 2A')

lieux = pd.read_csv('lieux-2021.csv',sep= ";")
usagers = pd.read_csv('usagers-2021.csv',sep= ";")
vehicules = pd.read_csv('vehicules-2021.csv', sep= ";")
caracteristiques = pd.read_csv("carcteristiques-2021.csv",sep = ";")


lieux.isnull().sum()
usagers.isnull().sum()
vehicules.isnull().sum()
caracteristiques.isnull().sum()

test = pd.merge(caracteristiques, lieux, on="Num_Acc")

#test2 = pd.concat([usagers,vehicules])

#test = test.loc[~test.index.duplicated(keep='first')]
#test2 = test2.loc[~test2.index.duplicated(keep='first')]
#test3 = pd.concat([test, test2], axis=1)

n,p = test.shape
m,q = test2.shape

#se merge pas bien : 129153 - 72635 = 56518

test.describe

test4 = test.align(test2)

valeurs = lieux['Num_Acc']
valeurs2 = valeurs.to_numpy()
repet = usagers['Num_Acc']
repet2 = repet.to_numpy()

valeurs3 = vehicules['id_vehicule']
valeurs4 = valeurs3.to_numpy()
repet3 = usagers['id_vehicule']
repet4 = repet3.to_numpy()

#but créer un df contenant les infos de test mais avec des lignes doublées en fn du nb de passagers dans la voiture

def compte(valeur,repet):
    #compte le nombre de répétitions de lignes pour un accident dans test2 
    L = []
    nn = len(valeur)
    for i in range(nn):
        a = valeur[i]
        c = np.count_nonzero(repet == a)
        L.append(c)
    return L
        

liste = compte(valeurs2,repet2)
# comment passer de 56518 à 129153

liste2 =compte(valeurs4,repet4)

fruit_list = [ ('Orange', 34, 'Yes' )]
df = pd.DataFrame(fruit_list, columns = ['Name' , 'Price', 'Stock'])

def creat(vehicules,liste):
    vehicules2 = vehicules.copy()
    l = len(liste)
    ll = 0
    for i in range(l):
        if liste[i] !=1:
            for j in range(liste[i]-1):
                vehicules2.loc[l+ll] = vehicules.iloc[i]
                ll+=1
    return vehicules2
            
#on crée les lignes doublons dans véhicules qui correspondent aux usagers d'un même véhicule afin de concat avec usagers ensuite par les colonnes
vehicules2 = creat(vehicules,liste2)

vehicules3 = vehicules2.sort_values(by = ['Num_Acc', 'id_vehicule'])    
    
#6 lignes de décalage dans les deux dataframes véhicules3 et usagers, jsp pq
vehicules4 = vehicules3.reset_index(drop=True)
    

bug = vehicules5['Num_Acc']
bug2 = bug.to_numpy()
bugprime = usagers['Num_Acc']
bugprime2 = bugprime.to_numpy()   
    

A = compte(valeurs2,bug2)
B = compte(valeurs2,bugprime2)

def diff(A,B):
    l = len(A)
    L = []
    for i in range(l):
        L.append(A[i]-B[i])
    return L

#lignes de vehicules4 à drop : 
    #première de Num_Acc = 202100009430
    #dernière de Num_Acc = 202100022987
    #dernière de Num_Acc = 202100040202
    #première de Num_Acc = 202100040933
    #première de Num_Acc = 202100044328
    #première de Num_Acc = 202100053167
    
    
vehicules5 = vehicules4.drop(index = [21787,52699,92651,94291,101765,121556])

test2 = pd.concat([vehicules5,usagers], axis = 1)

