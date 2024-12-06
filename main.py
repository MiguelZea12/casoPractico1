from data_loader import load_data, preprocess_data
from model import train_polynomial_regression, evaluate_model
from optimizer import optimize_budget
from visualizer import visualize_results

def main():
    filepath = "advertising.csv"
    df = load_data(filepath)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model, poly = train_polynomial_regression(X_train, y_train)
    train_rmse, test_rmse, r2, y_pred_test = evaluate_model(model, poly, X_train, X_test, y_train, y_test)

    print(f"RMSE (Entrenamiento): {train_rmse:.4f}")
    print(f"RMSE (Prueba): {test_rmse:.4f}")
    print(f"R^2 (Prueba): {r2:.4f}")

    optimized_budget, max_sales = optimize_budget(model, poly)
    print("\nPresupuesto óptimo por canal publicitario:")
    print(f"TV: ${optimized_budget[0]:.2f} mil")
    print(f"Radio: ${optimized_budget[1]:.2f} mil")
    print(f"Newspaper: ${optimized_budget[2]:.2f} mil")
    print(f"Ventas máximas esperadas: {max_sales:.4f} miles de unidades")

    visualize_results(df, y_test, y_pred_test, optimized_budget, poly, model, X_train)

if __name__ == "__main__":
    main()
