#
# Busque los mejores parametros de un modelo elasticnet para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Considere diferentes valores para la cantidad de vecinos
#

# importacion de librerias

from homework.src.internals.calculate_metrics import calculate_metrics
from homework.src.internals.prepare_data import prepare_data
from homework.src.internals.print_metrics import print_metrics
from homework.src.internals.save_model import save_model

def run_experiment(estimator):

    x_train,x_test, y_train,y_test = prepare_data()

    estimator.fit(x_train, y_train)
    save_model(estimator)
    
    print()
    print(estimator, ":", sep="")

    # Metricas de error durante entrenamiento
    mse, mae, r2 = calculate_metrics(x_train, y_train, estimator)
    print_metrics(mse, mae, r2, title = "Metricas de entrenamiento:")

    # Metricas de error durante testing
    mse, mae, r2 = calculate_metrics(x_test, y_test, estimator)
    print_metrics(mse, mae, r2, title = "Metricas de testing:")
