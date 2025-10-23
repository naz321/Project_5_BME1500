import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load features and label sheets
features = pd.read_excel("/Users/naziba/Desktop/Project_5_BME1500/shareable_dataset/neuron_features_with_lfp.xlsx")
labels = pd.read_excel("/Users/naziba/Desktop/Project_5_BME1500/shareable_dataset/bme1500-project-5-metadata.xlsx")

print(labels.columns)
print(features.columns)
data = pd.merge(features, labels, on="Filename")

print(data.head())

# Features we use to train the model
X = data[["FiringRate_Hz", "ISI_Mean_s", "CV_ISI", "BurstIndex", "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power", "total_power", "signal_rms", "variance"]]
# Label vector — what we are trying to predict
y = data["Target"]  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

clf = SVC(kernel='rbf', C=1, gamma='auto')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=clf.classes_,
            yticklabels=clf.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

