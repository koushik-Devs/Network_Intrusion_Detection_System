import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------- 1. Load Dataset ----------
train_data = pd.read_csv("data/KDDTrain+.txt", header=None)
test_data = pd.read_csv("data/KDDTest+.txt", header=None)

# ---------- 2. Assign Column Names ----------
col_names = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

train_data.columns = col_names
test_data.columns = col_names

# ---------- 3. Convert to Binary Label ----------
train_data['binary_label'] = train_data['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
test_data['binary_label'] = test_data['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')

# ---------- 4. Encode Categorical Features ----------
cat_cols = ['protocol_type', 'service', 'flag']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
    encoders[col] = le

# ---------- 5. Feature Selection ----------
X_train = train_data.drop(['label', 'difficulty', 'binary_label'], axis=1)
y_train = train_data['binary_label']
X_test = test_data.drop(['label', 'difficulty', 'binary_label'], axis=1)
y_test = test_data['binary_label']

# ---------- 6. Feature Scaling ----------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- 7. Train SVM Model ----------
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)

# ---------- 8. Save Trained Model ----------
joblib.dump(svm, 'models/svm_model.pkl')

# ---------- 9. Evaluate Model ----------
y_pred = svm.predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------- 10. Visualizations ----------

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", xticklabels=['normal', 'attack'], yticklabels=['normal', 'attack'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Top 10 Attacks (Original Labels)
top_attacks = train_data['label'].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_attacks.index, y=top_attacks.values, palette="viridis")
plt.title("Top 10 Most Frequent Attacks in Training Set")
plt.xlabel("Attack Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
