# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as pltų

data = pd.read_csv("boston_house_prices.csv")
#Récuperer les variable indépendentes explicatives (tous les var sauf MEDV)
X =data.drop("MEDV",axis=1)
#Récuperer var dépendante MEDV
y = data.MEDV

#Définir la taille de graphe
plt.figure(figsize=(75, 5))

#Faire une boucle pour afficher la var MEDV 'prix' par rapport chaque variable de x
for i, col in enumerate(X.columns):
    plt.subplot(1, 13, i+1)
    x = data[col]
    y = y
    plt.plot(x, y, 'o')
    #Création de la ligne de regression
    plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    plt.style.use(['dark_background','fast'])
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('prix')
#La résultat doit etre 15 graphes

#Fractionnement du dataset entre le Training set et le Test set (les données de test doit composés 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0)

scaler = StandardScaler()
scaler.fit(X_train)
scaler.fit(X_test)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Construction du modèle
regressor = LinearRegression()
#J'adapte le modèle de regression linéaire à l'ensemble des données d'apprentissage
regressor.fit(X_train, y_train)

#Faire de nouvelles prédictions
y_pred = regressor.predict(X_test)

#Faire comparer entre les var de prédictions et de test (voir est-ce qu'il ya la corrélation)
#plt.style("bmh")
plt.scatter(y_pred, y_test)
plt.show


#Faire des prédictions avec des valeurs aléatoires (c a d quelle est le prix si on a ses parametres)
regressor.predict(scaler.fit_transform(np.array([[0.17331,0,9.69,0,0.585,5.707,54,2.3817,6.391,19.2,396.9,12.01,0]])))







#================Evaluation et validation=================
#Définir constant de fonction
constante = regressor.intercept_
print(constante)

#Définir les coefficients de fonction
coefficients = regressor.coef_
print(coefficients)

#Définir les noms de champs
nom = [i for i in list (X)]
print(nom)

#Définr la somme des écarts entre la valeurs observés et les valeurs prédictés
erreur_quadratique_moyenne = np.mean((y_pred - y_test)**2)
print(erreur_quadratique_moyenne)

#Evaluer notre modèle
import statsmodels.api as sm
#Récupérer le résumé de modèle
model1 = sm.OLS(y_train, X_train)
result = model1.fit( )
print(result.summary())
#Il nous donne un tableau de bord statistique et on poura faire analyse à partir de ce tableau
# on a  R-squared (uncentered): 0.327 (plus il rapproche a 1 plus le modele est bien)
# on a la qualité AIC: 3594. (plus la qualité est faible plus le modèle est meilleure)



























