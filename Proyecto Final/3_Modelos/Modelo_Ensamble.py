import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Cargar datos
df = pd.read_csv('MutualFunds_Model_Ready.csv')

# Ajuste para XGBoost (clases deben empezar en 0)
df['morningstar_risk_rating'] = df['morningstar_risk_rating'] - 1

X = df.drop('morningstar_risk_rating', axis=1)
y = df['morningstar_risk_rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Definir los "Jugadores"
model_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model_xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1)

# 3. Crear el Ensamble (Voting)
# voting='soft' significa que promedian sus probabilidades (es más preciso que 'hard')
ensemble_model = VotingClassifier(
    estimators=[('rf', model_rf), ('xgb', model_xgb)],
    voting='soft' 
)

# 4. Entrenar el equipo completo
print("Entrenando el Ensamble (RF + XGBoost)...")
ensemble_model.fit(X_train, y_train)

# 5. Evaluar
y_pred = ensemble_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"--- RESULTADO ENSAMBLE ---")
print(f"Accuracy Combinado: {acc:.4%}")
print("\n", classification_report(y_test, y_pred))
