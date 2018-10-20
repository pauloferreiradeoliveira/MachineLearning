# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 10:54:21 2018

@author: paulo
"""

import pandas as pd
base = pd.read_csv('Dados/risco-credito.csv')

previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
previsores[:,0] = label.fit_transform(previsores[:,0])
previsores[:,1] = label.fit_transform(previsores[:,1])
previsores[:,2] = label.fit_transform(previsores[:,2])
previsores[:,3] = label.fit_transform(previsores[:,3])



from sklearn.naive_bayes import GaussianNB
clasificador = GaussianNB()

#Trenamento da probilidade
clasificador.fit(previsores,classe)

#HISTORIA: BOA,Divida: Alta, Garantias: nenhuma, Renda > 35
#HISTORIA: ruim ,Divida: Alta, Garantias: adquada, Renda < 15
resultador = clasificador.predict([[0,0,1,2],[3,0,0,0]])

print(clasificador.classes_)
print(clasificador.class_count_)
print(clasificador.class_prior_)
