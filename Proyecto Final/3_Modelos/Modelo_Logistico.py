import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Cargar datos listos
df = pd.read_csv('MutualFunds_Model_Ready.csv')

# 2. Separar X e y
X = df.drop('morningstar_risk_rating', axis=1)
y = df['morningstar_risk_rating']

# 3. Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Escalado (OBLIGATORIO para Regresión Logística)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Entrenar Modelo Lineal
print("Entrenando Regresión Logística...")
# max_iter=1000 para que le de tiempo a converger
lr_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# 6. Evaluar
y_pred = lr_model.predict(X_test_scaled)

print("--- RESULTADOS BASELINE (Regresión Logística) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nReporte de Clasificación:\n")
print(classification_report(y_test, y_pred))

s
