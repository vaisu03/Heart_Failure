# aiw_pso_gbm_type2.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from mealpy.swarm_based import PSO
from mealpy.utils.problem import Problem
import warnings
warnings.filterwarnings("ignore")
from mealpy.utils.space import IntegerVar, FloatVar

# Define bounds using mealpy variable types
bounds = [
    IntegerVar(lb=50, ub=300),      # n_estimators
    FloatVar(lb=0.01, ub=0.3),      # learning_rate
    IntegerVar(lb=3, ub=10)         # max_depth
]

problem = Problem(bounds=bounds, minmax="min", fit_func=fitness)


# =====================
# 1. Load dataset
# =====================
df = pd.read_csv("data/heart_failure.csv")

# Features & Target
X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

# =====================
# 2. Preprocessing
# =====================
# MinMaxScaler for chi2 (non-negative)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for later use in Streamlit
joblib.dump(scaler, "scaler.pkl")

# SMOTE for balancing
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

# Select Top 7 Features
selector = SelectKBest(score_func=chi2, k=7)
X_selected = selector.fit_transform(X_res, y_res)
selected_feature_names = X.columns[selector.get_support()]
print("✅ Selected Features:", selected_feature_names.tolist())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_res, test_size=0.2, random_state=42)

# =====================
# 3. AIW-PSO Optimization
# =====================
def fitness(solution):
    n_estimators = int(solution[0])
    learning_rate = solution[1]
    max_depth = int(solution[2])
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
    return 1 - scores.mean()  # Minimize error

# Define bounds for hyperparameters
bounds = [
    (50, 300),      # n_estimators
    (0.01, 0.3),    # learning_rate
    (3, 10)         # max_depth
]

problem = Problem(bounds=bounds, minmax="min", fit_func=fitness)

print("🔍 Running AIW-PSO Optimization...")
model_pso = PSO.OriginalPSO(epoch=30, pop_size=10, c1=2.05, c2=2.05, w=0.4)
best_solution = model_pso.solve(problem)

best_n, best_lr, best_depth = int(best_solution.solution[0]), best_solution.solution[1], int(best_solution.solution[2])
print(f"🏆 Best Hyperparameters: n_estimators={best_n}, learning_rate={best_lr}, max_depth={best_depth}")

# =====================
# 4. Train Final GBM Model
# =====================
final_model = LGBMClassifier(n_estimators=best_n, learning_rate=best_lr, max_depth=best_depth, random_state=42)
final_model.fit(X_train, y_train)

# Save model for Streamlit
joblib.dump(final_model, "heart_failure_model.pkl")
joblib.dump(selected_feature_names.tolist(), "selected_features.pkl")

# =====================
# 5. Evaluation
# =====================
y_pred = final_model.predict(X_test)

print("\n📊 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))

print("✅ Training Complete & Models Saved Successfully!")
