import pandas as pd
import numpy as np
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, cross_val_score
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ===============================
# STEP 1: Load & Process Dataset
# ===============================
df = pd.read_csv('data/heart_failure.csv')

# ✅ Log-transform skewed features
df['creatinine_phosphokinase'] = np.log1p(df['creatinine_phosphokinase'])
df['serum_creatinine'] = np.log1p(df['serum_creatinine'])

# ✅ Minimal feature engineering (keep only useful)
df['creatinine_sodium_ratio'] = df['serum_creatinine'] / df['serum_sodium']

# ✅ Ensure binary columns are integers
binary_cols = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
df[binary_cols] = df[binary_cols].astype(int)

# ✅ Features & Target
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# ✅ Remove low-variance features
vt = VarianceThreshold(threshold=0.01)
X = pd.DataFrame(vt.fit_transform(X), columns=X.columns[vt.get_support()])

# ✅ Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Balance dataset (use SMOTE instead of SMOTETomek)
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)
print("✅ Dataset processed:", X_res.shape, "samples ready")

# ===============================
# STEP 2: Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# ===============================
# STEP 3: Custom AIW-PSO for GBM
# ===============================
def fitness_gbm(params):
    n_estimators = int(params[0])
    learning_rate = params[1]
    max_depth = int(params[2])
    
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    return 1 - scores.mean()

def aiw_pso(fitness_func, bounds, num_particles=30, max_iter=50, w_max=0.9, w_min=0.4, c1=2, c2=2):
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    X = np.random.uniform(lb, ub, (num_particles, dim))
    V = np.zeros((num_particles, dim))
    pbest = X.copy()
    pbest_score = np.array([fitness_func(x) for x in X])
    gbest_idx = np.argmin(pbest_score)
    gbest = pbest[gbest_idx].copy()
    gbest_score = pbest_score[gbest_idx]
    
    for t in range(max_iter):
        w = w_max - ((w_max - w_min) * t / max_iter)
        for i in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (gbest - X[i])
            X[i] = np.clip(X[i] + V[i], lb, ub)
            score = fitness_func(X[i])
            if score < pbest_score[i]:
                pbest[i], pbest_score[i] = X[i].copy(), score
                if score < gbest_score:
                    gbest, gbest_score = X[i].copy(), score
        print(f"Iteration {t+1}/{max_iter}, Best CV Accuracy: {1-gbest_score:.4f}")
    return gbest, 1 - gbest_score

# ===============================
# STEP 4: Run AIW-PSO Optimization
# ===============================
bounds = [
    (50, 500),    # n_estimators
    (0.005, 0.3), # learning_rate
    (2, 12)       # max_depth
]

best_pos, best_acc = aiw_pso(fitness_gbm, bounds)
print("✅ Best Hyperparameters:", best_pos, "CV Accuracy:", best_acc)

# Train final GBM
best_n, best_lr, best_depth = int(best_pos[0]), best_pos[1], int(best_pos[2])
final_model = LGBMClassifier(n_estimators=best_n, learning_rate=best_lr, max_depth=best_depth, random_state=42)
final_model.fit(X_train, y_train)

# ===============================
# STEP 5: Evaluation
# ===============================
y_pred = final_model.predict(X_test)
print("\n📊 Optimized GBM Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_pred))

# ROC Curve
y_prob = final_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'Optimized GBM (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Feature Importance
plt.figure(figsize=(8,5))
sns.barplot(x=final_model.feature_importances_, y=X.columns, legend=False, palette="viridis")
plt.title("Feature Importance (Optimized GBM)")
plt.show()

# Save Model
joblib.dump(final_model, "heart_failure_model.pkl")
print("✅ Training Complete & Optimized GBM saved as heart_failure_model.pkl")
