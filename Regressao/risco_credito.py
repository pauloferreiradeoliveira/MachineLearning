
import pandas as pd
base = pd.read_csv('risco-credito.csv')

previsores = base.iloc[:,0:4].values
classe = base.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
previsores[:,0] = label.fit_transform(previsores[:,0])
previsores[:,1] = label.fit_transform(previsores[:,1])
previsores[:,2] = label.fit_transform(previsores[:,2])
previsores[:,3] = label.fit_transform(previsores[:,3])

from sklearn.linear_model import LogisticRegression
classiicador = LogisticRegression()
classiicador.fit(previsores,classe)
print(classiicador.intercept_)
print(classiicador.coef_)

#HISTORIA: BOA,Divida: Alta, Garantias: nenhuma, Renda > 35
#HISTORIA: ruim ,Divida: Alta, Garantias: adquada, Renda < 15
resultador = classiicador.predict([[0,0,1,2],[3,0,0,0]])
resultador2 = classiicador.predict_proba([[0,0,1,2],[3,0,0,0]])
