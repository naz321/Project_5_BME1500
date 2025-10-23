# train_logreg_with_lfp.py
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
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

# ---------- Impute missing numeric values ----------
imp = SimpleImputer(strategy="median")
X = pd.DataFrame(imp.fit_transform(X_raw), columns=feature_cols)

# ---------- Generic trainer ----------
def run_logreg(label_col):
    print(f"\n=== Logistic Regression for {label_col.upper()} ===")

    # encode labels
    y = df[label_col].astype(str)
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)

    # safe stratified split
    vc = pd.Series(y_enc).value_counts()
    strat = y_enc if vc.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=strat, random_state=42
    )

    # build model: scale + multinomial logistic regression
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000
        )
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, target_names=enc.classes_))
    return acc

# ---------- Run for both tasks ----------
acc_target = run_logreg("target")
acc_neuron = run_logreg("neuron")

print("\nSummary:")
print(f"Target accuracy: {acc_target:.3f}")
print(f"Neuron accuracy: {acc_neuron:.3f}")