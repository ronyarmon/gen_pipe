from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.svm import LinearSVR, SVR

models = [LinearRegression(), SGDRegressor(),LinearSVR(), SVR()]
for model in models:
    class_name = model.__class__.__name__
    print(class_name, type(class_name))
    print(model.get_params())
    print(model._get_param_names())
