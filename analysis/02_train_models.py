import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

df = pd.read_csv("../data/processed/features.csv")

if len(df) == 0:
    print("❌ Dataset features.csv masih kosong. Jalankan 01_extract_features.py terlebih dahulu.")
    exit(1)

features = ['num_samples', 'num_unique', 'dispersion', 'opt1', 'opt2', 'opt3', 'all_opts', 'sentence']
X = df[features]
y = df['label_tut']
groups = df['participant_id']

print(f"📊 Ringkasan Dataset:")
print(f"   - Total Baris Data : {len(df)}")
print(f"   - Total Partisipan : {df['participant_id'].nunique()}")
print(f"   - Distribusi Label :")
print(f"     * Label 0 (On-Task / Fokus)    : {sum(y == 0)}")
print(f"     * Label 1 (Off-Task / Melamun) : {sum(y == 1)}")

if y.nunique() < 2:
    print("\n⚠️ PERHATIAN:")
    print("   Target 'label_tut' baru memiliki 1 kelas saja!")
    print("   Model klasifikasi dan SMOTE membutuhkan MINIMAL 2 KELAS (ada data berlabel 0 dan data berlabel 1).")
    print("   👉 Penyebab: Kedua partisipan sebelumnya memilih 'Tidak (On-Task / Fokus)'.")
    print("   👉 Solusi: Lakukan eksperimen lagi dan pilih 'Ya (Off-Task / Melamun)' pada modal evaluasi fokus.")
    exit(0)

n_participants = df['participant_id'].nunique()
if n_participants < 2:
    print("\n⚠️ Jumlah partisipan kurang dari 2. GroupKFold membutuhkan minimal 2 partisipan.")
    exit(0)

n_splits = min(10, n_participants)
gkf = GroupKFold(n_splits=n_splits)

y_true_all, y_pred_all = [], []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Standardize (Hanya Train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Cek apakah train set memiliki kedua kelas
    if y_train.nunique() < 2:
        print(f"⚠️ [Fold {fold}] Data training hanya memiliki 1 kelas ({y_train.iloc[0]}). Butuh lebih banyak partisipan dengan variasi label.")
        continue

    # SMOTE Adaptif sesuai jumlah sampel kelas minoritas
    min_class_count = y_train.value_counts().min()
    if min_class_count >= 2:
        k_neighbors = min(5, min_class_count - 1)
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    else:
        # Jika sampel kelas minoritas hanya 1, lewati SMOTE untuk menghindari error k-neighbors
        X_train_res, y_train_res = X_train_scaled, y_train

    # Model XGBoost persis paper
    clf = XGBClassifier(n_estimators=100, max_depth=3, eval_metric='logloss', random_state=42)
    clf.fit(X_train_res, y_train_res)

    preds = clf.predict(X_test_scaled)
    y_true_all.extend(y_test)
    y_pred_all.extend(preds)

if len(y_true_all) == 0:
    print("\n❌ Belum ada fold yang dapat dievaluasi. Kumpulkan data dari lebih banyak partisipan dengan label bervariasi.")
    exit(0)

print("\n=== HASIL EVALUASI PREDIKSI MIND WANDERING (TUT) ===")
print(f"Cohen's Kappa : {cohen_kappa_score(y_true_all, y_pred_all):.3f}")
print(f"Weighted F1   : {f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0):.3f}")
print(f"F1 (TUT Class): {f1_score(y_true_all, y_pred_all, pos_label=1, zero_division=0):.3f}")
print(f"Precision TUT : {precision_score(y_true_all, y_pred_all, pos_label=1, zero_division=0):.3f}")
print(f"Recall TUT    : {recall_score(y_true_all, y_pred_all, pos_label=1, zero_division=0):.3f}")
