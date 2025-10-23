# train_knn_with_lfp.py
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# ---------- Load ----------
df = pd.read_csv('neurons/myNeuronFeatures_wLFP.csv')

# ---------- Feature list ----------
feature_cols = [
    # spike / burst
    "firing_rate","burst_index","cv",
    "burst_dur_theta","burst_dur_alpha","burst_dur_lowbeta","burst_dur_highbeta",
    "spike_power_theta","spike_power_alpha","spike_power_lowbeta","spike_power_highbeta",
    # LFP absolute powers
    "lfp_theta_power","lfp_alpha_power","lfp_lowbeta_power","lfp_highbeta_power",

]
feature_cols = [c for c in feature_cols if c in df.columns]
X_raw = df[feature_cols].copy()

# ---------- Impute (keep rows; fill gaps) ----------
imp = SimpleImputer(strategy="median")
X = pd.DataFrame(imp.fit_transform(X_raw), columns=feature_cols)

# ---------- Helper: pick a simple k ----------
def pick_k(n_train):
    k = int(np.sqrt(max(1, n_train)))
    if k % 2 == 0: k += 1     # prefer odd to reduce ties
    return max(1, min(k, n_train))

# ---------- Train/eval one KNN task ----------
def run_knn(label_col):
    print(f"\n=== KNN for {label_col.upper()} ===")
    y = df[label_col].astype(str)

    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)

    # safe stratify if all classes have >=2 samples
    vc = pd.Series(y_enc).value_counts()
    strat = y_enc if vc.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=strat, random_state=42
    )

    k = pick_k(len(X_train))

    model = make_pipeline(
        StandardScaler(),                      # KNN needs scaling
        KNeighborsClassifier(
            n_neighbors=k,
            weights="distance",                # helps with class imbalance
            metric="minkowski", p=2            # Euclidean
        )
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"k = {k} | Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=enc.classes_))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=enc.classes_, yticklabels=enc.classes_)
    plt.title("Confusion Matrix — Neuron Classification")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    return acc



# ---------- Run both tasks ----------
acc_target = run_knn("target")
acc_neuron = run_knn("neuron")

print("\nSummary:")
print(f"Target accuracy: {acc_target:.3f}")
print(f"Neuron accuracy: {acc_neuron:.3f}")


