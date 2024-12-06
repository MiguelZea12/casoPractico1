import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_results(df, y_test, y_pred_test, optimized_budget, poly, model, X_train):
    """Genera visualizaciones de los resultados."""
    # Influencia de los Canales Publicitarios
    coefs = model.coef_
    features = poly.get_feature_names_out(X_train.columns)
    importance = pd.DataFrame({"Feature": features, "Coefficient": coefs})

    # Crear subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Relación entre inversión publicitaria y ventas
    sns.scatterplot(ax=axs[0, 0], data=df, x='TV', y='Sales', alpha=0.7, label='TV')
    sns.scatterplot(ax=axs[0, 0], data=df, x='Radio', y='Sales', alpha=0.7, label='Radio')
    sns.scatterplot(ax=axs[0, 0], data=df, x='Newspaper', y='Sales', alpha=0.7, label='Newspaper')
    axs[0, 0].set_title("Relación entre Canales Publicitarios y Ventas")
    axs[0, 0].set_xlabel("Inversión Publicitaria (miles de $)")
    axs[0, 0].set_ylabel("Ventas (miles de unidades)")
    axs[0, 0].legend()

    # Influencia de los Canales Publicitarios
    sns.barplot(ax=axs[0, 1], data=importance, x='Feature', y='Coefficient', palette='viridis')
    axs[0, 1].set_title("Influencia de los Canales Publicitarios (Coeficientes)")
    axs[0, 1].set_xticklabels(importance['Feature'], rotation=45)

    # Presupuesto Óptimo por Canal
    optimized_df = pd.DataFrame({
        "Canal": ["TV", "Radio", "Newspaper"],
        "Presupuesto": optimized_budget
    })
    sns.barplot(ax=axs[1, 0], data=optimized_df, x='Canal', y='Presupuesto', palette='coolwarm')
    axs[1, 0].set_title("Presupuesto Óptimo por Canal Publicitario")
    axs[1, 0].set_ylabel("Presupuesto (miles de dólares)")

    # Comparación entre predicciones y valores reales
    axs[1, 1].scatter(y_test, y_pred_test, alpha=0.7, label='Predicciones')
    axs[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Línea Ideal')
    axs[1, 1].set_title("Predicciones vs Ventas Reales")
    axs[1, 1].set_xlabel("Ventas Reales (miles de unidades)")
    axs[1, 1].set_ylabel("Ventas Predichas (miles de unidades)")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()
