#
# Busque los mejores parametros de un modelo elasticnet para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Considere diferentes valores para la cantidad de vecinos
#

# importacion de librerias
from sklearn.linear_model import ElasticNet
from homework.src.internals.calculate_metrics import calculate_metrics
from homework.src.internals.prepare_data import prepare_data
from homework.src.internals.print_metrics import print_metrics
from homework.src.internals.save_model import save_model

x_train, x_test, y_train, y_test = prepare_data()


# entrenar el modelo
estimator = ElasticNet(alpha=0.5,l1_ratio=0.5,random_state=12345,)
estimator.fit(x_train, y_train)
save_model(estimator)

print()
print(estimator, ":", sep="")

mse, mae, r2 = calculate_metrics(x_train, y_train, estimator)
print_metrics(mse, mae, r2, title="Metricas de entrenamiento:")

mse, mae, r2 = calculate_metrics(x_test, y_test, estimator)
print_metrics(mse, mae, r2, title="Metricas de testing:")

