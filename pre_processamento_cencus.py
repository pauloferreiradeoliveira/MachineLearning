import pandas as pd

base = pd.read_csv('census.csv')
previsores = base.iloc[:,0:14].values
classe =base.iloc[:,14].values

# trasformação string em numeros
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_previsores = LabelEncoder()

#labels = labelencoder_previsores.fit_transform(previsores[:,1])
#previsores[:,0] = labelencoder_previsores.fit_transform(previsores[:,0])

previsores[:,1] = labelencoder_previsores.fit_transform(previsores[:,1])
previsores[:,3] = labelencoder_previsores.fit_transform(previsores[:,3])
previsores[:,5] = labelencoder_previsores.fit_transform(previsores[:,5])
previsores[:,6] = labelencoder_previsores.fit_transform(previsores[:,6])
previsores[:,7] = labelencoder_previsores.fit_transform(previsores[:,7])
previsores[:,8] = labelencoder_previsores.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder_previsores.fit_transform(previsores[:,9])
previsores[:,13] = labelencoder_previsores.fit_transform(previsores[:,13])

onehotcoder = OneHotEncoder(categorical_features=[1,3,5,6,7,8,9,13])
previsores = onehotcoder.fit_transform(previsores).toarray()

#from sklearn.compose import ColumnTransformer, make_column_transformer

#preprocess = make_column_transformer( ([1,3,5,6,7,8,9,13],OneHotEncoder()))
#preprocess.remainder = 'passthrough'
#previsores = preprocess.fit_transform(previsores).toarray()

labelencder_classe = LabelEncoder()
classe = labelencder_classe.fit_transform(classe)

from sklearn.preprocessing import StandardScaler
sclar = StandardScaler()
previsores = sclar.fit_transform(previsores)
