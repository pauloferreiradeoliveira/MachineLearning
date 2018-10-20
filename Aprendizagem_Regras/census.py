import Orange

base = Orange.data.Table('Dados/census.csv')

base_dividida = Orange.evaluation.testing.sample(base,n=0.15)
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

cn2_learner = Orange.classification.rules.CN2Learner()
classificador = cn2_learner(base_treinamento)
