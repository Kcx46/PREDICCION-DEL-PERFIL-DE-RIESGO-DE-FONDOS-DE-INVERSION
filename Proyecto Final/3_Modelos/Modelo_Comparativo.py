import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------
# 1. CARGAR RESULTADOS (Manual o desde archivo)
# -----------------------------------------------------------
# Aquí pones los números finales que obtuviste en los notebooks anteriores
# (Pon los valores reales que te salieron)

resultados = {
    'Modelo': [
        'Regresión Logística', 
        'XGBOOST', 
        'Ensamble',
        'Random Forest'
    ],
    'Tipo': [
        'Lineal (Base)', 
        'No Lineal (Boosting)', 
        'No Lineal (Stacking)',
        'No Lineal (Bagging)'
    ],
    # Nota: Ordené los modelos para que el mejor (RF) salga al final o se destaque
    'Accuracy': [0.5173, 0.7604, 0.8722, 0.8934],
    'Comentarios': [
        'No captura complejidad', 
        'Mejora patrones no lineales', 
        'Gran estabilidad',
        'Mejor desempeño general'
    ]
}

df_resultados = pd.DataFrame(resultados)

# -----------------------------------------------------------
# 2. GUARDAR CSV FINAL
# -----------------------------------------------------------
# Esto cumple con el entregable 'resultados_modelos.csv' de su estructura
df_resultados.to_csv('resultados_modelos.csv', index=False)
print("Archivo 'resultados_modelos.csv' generado.")
print(df_resultados)

# -----------------------------------------------------------
# 3. GRAFICAR LA COMPARACIÓN (La "Slide Ganadora")
# -----------------------------------------------------------
plt.figure(figsize=(8, 6))

# Gráfico de barras
ax = sns.barplot(x='Modelo', y='Accuracy', data=df_resultados, palette=['gray', 'green'])

# Poner los números encima de las barras
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2%}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points',
                fontsize=12, fontweight='bold')

plt.ylim(0, 1.0) # Eje Y de 0 a 100%
plt.title('Comparación de Desempeño: Lineal vs No Lineal', fontsize=14)
plt.ylabel('Exactitud (Accuracy)')
plt.xlabel('')

# Guardar para el reporte
plt.savefig('Resultados/Comparacion_Modelos.png', dpi=300, bbox_inches='tight')
plt.show()
