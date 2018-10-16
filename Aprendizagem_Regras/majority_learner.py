import Orange

base = Orange.data.Table('credit-data.csv')

base_dividida = Orange.evaluation.testing.sample(base,n=0.25)
base_treinamento = base_dividida[1];
base_test = base_dividida[0]


classificador = Orange.classification.MajorityLearner()
resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento,base_test,[classificador])

print(Orange.evaluation.CA(resultado))
