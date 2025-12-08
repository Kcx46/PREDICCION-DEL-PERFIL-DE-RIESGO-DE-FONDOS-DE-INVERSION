import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Crear carpeta de resultados si no existe (opcional, para orden)
if not os.path.exists('Resultados'):
    os.makedirs('Resultados')

# 1. Cargar datos listos
df = pd.read_csv('MutualFunds_Model_Ready.csv')

X = df.drop('morningstar_risk_rating', axis=1)
y = df['morningstar_risk_rating']

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Entrenar Random Forest
print("Entrenando Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 4. Evaluar
y_pred = rf_model.predict(X_test)

print("--- RESULTADOS RANDOM FOREST ---")
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# -------------------------------------------------------
# 5. MATRIZ DE CONFUSIÓN
# -------------------------------------------------------
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title(f'Matriz de Confusión RF (Acc: {acc:.2%})')
plt.ylabel('Real')
plt.xlabel('Predicho')

# GUARDAR LA IMAGEN
nombre_archivo_cm = 'Resultados/Matriz_Confusion_RF.png'
plt.savefig(nombre_archivo_cm, dpi=300, bbox_inches='tight')
print(f"Imagen guardada: {nombre_archivo_cm}")

plt.show()

# -------------------------------------------------------
# 6. FEATURE IMPORTANCE 
# -------------------------------------------------------
importances = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': rf_model.feature_importances_
}).sort_values('Importancia', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importancia', y='Variable', data=importances.head(10), palette='viridis')
plt.title('Top 10 Variables más importantes')

# GUARDAR LA IMAGEN
nombre_archivo_feat = 'Resultados/Importancia_Variables.png'
plt.savefig(nombre_archivo_feat, dpi=300, bbox_inches='tight')
print(f"Imagen guardada: {nombre_archivo_feat}")

plt.show()
