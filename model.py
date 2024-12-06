from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_polynomial_regression(X_train, y_train, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_train = poly.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_poly_train, y_train)
    return model, poly

def evaluate_model(model, poly, X_train, X_test, y_train, y_test):
    X_poly_train = poly.transform(X_train)
    X_poly_test = poly.transform(X_test)
    y_pred_train = model.predict(X_poly_train)
    y_pred_test = model.predict(X_poly_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)

    return train_rmse, test_rmse, r2, y_pred_test
