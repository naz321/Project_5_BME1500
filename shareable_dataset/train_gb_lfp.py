# train_gb_with_lfp.py
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

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

# ---------- Impute (keep data; fill gaps) ----------
imp = SimpleImputer(strategy="median")
X = pd.DataFrame(imp.fit_transform(X_raw), columns=feature_cols)

# ---------- Helper to train/evaluate one GB model ----------
def run_gb(label_col):
    print(f"\n=== Gradient Boosting for {label_col.upper()} ===")

    # encode labels
    y = df[label_col].astype(str)
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)

    # safe stratify
    vc = pd.Series(y_enc).value_counts()
    strat = y_enc if vc.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=strat, random_state=42
    )

    # class imbalance → sample weights (inverse frequency on train set)
    counts = pd.Series(y_train).value_counts()
    invfreq = {cls: 1.0/c for cls, c in counts.items()}
    sample_weight = np.array([invfreq[c] for c in y_train], dtype=float)

    # model (simple, robust defaults + early stopping)
    gb = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        validation_fraction=0.2,
        n_iter_no_change=10,
        random_state=0,
        tol=1e-4
    )

    gb.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred = gb.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=enc.classes_))

    # top features
    importances = gb.feature_importances_
    order = np.argsort(importances)[::-1][:10]
    print("Top features:")
    for i, j in enumerate(order, 1):
        print(f"{i:>2}. {feature_cols[j]:<22} {importances[j]:.4f}")

    return acc

# ---------- Run both tasks ----------
acc_target = run_gb("target")
acc_neuron = run_gb("neuron")

print("\nSummary:")
print(f"Target accuracy: {acc_target:.3f}")
print(f"Neuron accuracy: {acc_neuron:.3f}")