# multiclass_model.py
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

# Načítanie a príprava dát
data = pd.read_csv("preprocessed_data.csv").drop_duplicates()
le = LabelEncoder()
data["userid_encoded"] = le.fit_transform(data["userid"])
X = data.drop(["userid", "userid_encoded"], axis=1)
y = data["userid_encoded"]

# Rozdelenie dát
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Pipeline pre multi-triednu klasifikáciu
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('classifier', XGBClassifier(
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    ))
])

# Hyperparametre
param_dist = {
    'classifier__n_estimators': randint(200, 2000),
    'classifier__max_depth': randint(3, 15),
    'classifier__learning_rate': uniform(0.005, 0.3),
    'classifier__subsample': uniform(0.5, 0.5),
    'classifier__colsample_bytree': uniform(0.5, 0.5),
}

# Trénovanie modelu
random_search = RandomizedSearchCV(
    pipeline, param_dist, n_iter=100, cv=5,
    scoring='accuracy', n_jobs=-1, random_state=42
)

start_time = time.time()
random_search.fit(X_train, y_train)
print(f"Čas trénovania: {(time.time() - start_time)/60:.2f} min")
print(f"Najlepšie parametre: {random_search.best_params_}")

# Najlepší model
best_model = random_search.best_estimator_

# Predikcia a de-kódovanie
y_pred_encoded = best_model.predict(X_test)
y_pred = le.inverse_transform(y_pred_encoded)
y_true = le.inverse_transform(y_test)

# Vyhodnotenie
print(f"\nPresnosť: {accuracy_score(y_true, y_pred):.4f}")
y_proba = best_model.predict_proba(X_test)
print(f"ROC AUC (OvR): {roc_auc_score(y_test, y_proba, multi_class='ovr'):.4f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=le.classes_)
print("\nConfusion Matrix:")
print(cm)

# Vizualizácia
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Multiclass Model')
plt.xlabel('Predikovaný používateľ')
plt.ylabel('Skutočný používateľ')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Uloženie modelu a LabelEncoderu
joblib.dump(best_model, "multiclass_model.pkl")
joblib.dump(le, "label_encoder.pkl")
