import Orange

base = Orange.data.Table('credit-data.csv')

base_dividida = Orange.evaluation.testing.sample(base,n=0.25)
base_treinamento = base_dividida[1];
base_test = base_dividida[0]

cn2_learner = Orange.classification.rules.CN2Learner()
classificador = cn2_learner(base_treinamento)

for regras in classificador.rule_list:
    print(regras)

resultado = Orange.evaluation.testing.TestOnTestData(base_treinamento,base_test,[classificador])
print(Orange.evaluation.CA(resultado))
