import pandas as pd

base = pd.read_csv('Dados/credit-data.csv')
base.loc[base.age < 0, 'age'] = 40.92

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

import numpy as np
from sklearn.impute import SimpleImputer
simpleimputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer = simpleimputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento,previsores_teste,classe_treinamento,classe_teste = train_test_split(previsores,classe,test_size=0.25,random_state=0)

from sklearn.svm import SVC
classificador = SVC(kernel='rbf',random_state = 1,C = 2)
classificador.fit(previsores_treinamento,classe_treinamento)

# Salvar o CLassificador
from sklearn.externals import joblib
joblib.dump(classificador,'test.pk')

# Carregando o Classificador
test = joblib.load('test.pk')

resultado = classificador.predict(previsores_teste)


from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste,resultado)
matriz = confusion_matrix(classe_teste,resultado)
