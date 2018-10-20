import pandas as pd
base = pd.read_excel('alunos.xlsx')

previsores = base.iloc[:, 2:9].values
classe = base.iloc[:, 9].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder = LabelEncoder()
previsores[:,0] = labelEncoder.fit_transform(previsores[:, 0])
previsores[:,1] = labelEncoder.fit_transform(previsores[:, 1])
previsores[:,2] = labelEncoder.fit_transform(previsores[:, 2])
previsores[:,3] = labelEncoder.fit_transform(previsores[:, 3])
previsores[:,4] = labelEncoder.fit_transform(previsores[:, 4])
previsores[:,5] = labelEncoder.fit_transform(previsores[:, 5])
previsores[:,6] = labelEncoder.fit_transform(previsores[:, 6])


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.model_selection import train_test_split
previsores_treinamento,previsores_teste,classe_treinamento,classe_teste = train_test_split(previsores,classe,test_size=0.15,random_state=0)


from sklearn.svm import SVC
classificador = SVC(kernel='rbf', random_state = 1, C=2)
classificador.fit(previsores_treinamento,classe_treinamento)
previsoes = classificador.predict(previsores_teste)

#0.724
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste,previsoes)
matriz = confusion_matrix(classe_teste,previsoes)


#0.692
import collections
collections.Counter(classe)
