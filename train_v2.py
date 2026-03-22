import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle

print("Memulai Eksperimen Model Teroptimasi...")

# 1. Load Data
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset dimuat.")
except Exception as e:
    print(f"Error muat data: {e}")
    exit(1)

# 2. Preprocessing Cepat
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
X = df[['tenure', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn'].map({'Yes': 1, 'No': 0})

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Handling Imbalance dengan SMOTE
print("Menerapkan SMOTE untuk menyeimbangkan data...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"   Sebelum SMOTE: {np.bincount(y_train)}")
print(f"   Sesudah SMOTE: {np.bincount(y_train_res)}")

# 5. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# 6. Training dengan XGBoost
print("Melatih model XGBoost...")
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train_scaled, y_train_res)

# 7. Evaluasi
y_pred = model.predict(X_test_scaled)
print("\nHASIL EVALUASI MODEL BARU (XGBoost + SMOTE):")
print(classification_report(y_test, y_pred))

# 8. Simpan Model v2
print("Menyimpan model ke model_churn_v2.pkl...")
pickle.dump(model, open('model_churn_v2.pkl', 'wb'))
pickle.dump(scaler, open('scaler_v2.pkl', 'wb'))

print("\nSelesai! Model v2 berhasil dibuat.")
