import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from xgboost import XGBClassifier
import numpy as np

# 1. Cargar datos
df = pd.read_csv('MutualFunds_Model_Ready.csv')

# Ajustar el target para que empiece en 0 (XGBoost requiere clases 0, 1, 2, 3, 4)
# Morningstar es 1-5, así que restamos 1.
df['morningstar_risk_rating'] = df['morningstar_risk_rating'] - 1

X = df.drop('morningstar_risk_rating', axis=1)
y = df['morningstar_risk_rating']

# 2. Split (mismo random_state para comparar justamente)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Entrenar XGBoost
print("Entrenando XGBoost... (esto es rápido)")
xgb_model = XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=5, 
    objective='multi:softprob', 
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

# 4. Evaluar
y_pred = xgb_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"--- RESULTADOS XGBOOST ---")
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# -----------------------------------------------------------
# 5. GENERAR CURVA ROC MULTICLASE (Lo que les faltaba)
# -----------------------------------------------------------
# Binarizar las etiquetas para ROC (One-vs-Rest)
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
y_score = xgb_model.predict_proba(X_test)

# Calcular ROC y AUC para cada clase
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_bin.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Graficar
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'orange', 'purple']
clases_reales = [1, 2, 3, 4, 5] # Etiquetas originales para la leyenda

for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Clase {clases_reales[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC Multiclase (XGBoost)')
plt.legend(loc="lower right")

# Guardar la imagen faltante
plt.savefig('Resultados/ROC_multiclase.png', dpi=300, bbox_inches='tight')
print("'ROC_multiclase.png' guardada exitosamente.")
plt.show()
