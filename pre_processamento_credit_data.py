# -*- coding: utf-8 -*-

import pandas as pd

base = pd.read_csv('credit-data.csv')
base.describe()

# -- Valores Inconsistentes -- 
# Mostrando valores menor que 0
base.loc[base['age'] < 0]

# apagar a coluna
base.drop('age',1,inplace=True)

# apagar somento os registros com problema
base.drop(base[base.age < 0].index, inplace = True)

# preencher os valores manualmente
# preencher os valore com a media
base.mean()
base['age'].mean()

# media sem os negativos
base['age'][base.age > 0].mean()
base.loc[base.age < 0,'age'] = 40.92

# Valores Nulos
pd.isnull(base['age'])
base.loc[pd.isnull(base['age'])]

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# Deprecated 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean', axis= 0)
imputer = imputer.fit(previsores[:, 0:3])
previsores[:,0:3] = imputer.transform(previsores[:,0:3])

# Nova forma
import numpy as np
from sklearn.impute import SimpleImputer
simpleimputer = SimpleImputer(missing_values=np.nan,strategy='mean')
simpleimputer = simpleimputer.fit(previsores[:, 0:3])
previsores[:,0:3] = simpleimputer.transform(previsores[:,0:3])

# Escalonamento - Colocar na mesma Escalar
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)