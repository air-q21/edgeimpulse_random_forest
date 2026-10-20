import argparse
import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def main():
    parser = argparse.ArgumentParser(description='Train RAM-Optimized Random Forest')
    parser.add_argument('--data-directory', type=str, required=True)
    parser.add_argument('--out-directory', type=str, required=True)
    parser.add_argument('--n_estimators', type=int, default=25)  # Diturunkan dari 100 ke 25 agar hemat RAM
    parser.add_argument('--max_depth', type=int, default=8)       # Batasi kedalaman pohon
    args, _ = parser.parse_known_args()

    print(f"--> Konfigurasi: n_estimators={args.n_estimators}, max_depth={args.max_depth}", flush=True)

    x_train_path = os.path.join(args.data_directory, 'X_split_train.npy')
    y_train_path = os.path.join(args.data_directory, 'Y_split_train.npy')

    if not os.path.exists(x_train_path) or not os.path.exists(y_train_path):
        print(f"ERROR: Dataset tidak ditemukan di {args.data_directory}", flush=True)
        sys.exit(1)

    # 1. Konversi data ke float32 untuk memotong penggunaan memori hingga 50%
    X_train = np.load(x_train_path).astype(np.float32)
    Y_train = np.load(y_train_path)

    if len(X_train.shape) > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)

    if Y_train.ndim > 1:
        y_labels = Y_train[:, 0].astype(int)
    else:
        y_labels = Y_train.astype(int)

    print(f"--> Total sampel: {X_train.shape[0]}, Jumlah Fitur: {X_train.shape[1]}", flush=True)

    # 2. Gunakan n_jobs=1 agar Docker tidak membuat kloning memori multi-process
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features='sqrt',
        random_state=42,
        n_jobs=1
    )

    print("--> Memulai pelatihan Random Forest...", flush=True)
    clf.fit(X_train, y_labels)
    print("--> Pelatihan selesai!", flush=True)

    # 3. Simpan artefak model.pkl
    os.makedirs(args.out_directory, exist_ok=True)
    model_path = os.path.join(args.out_directory, 'model.pkl')
    joblib.dump(clf, model_path)
    print(f"--> Model berhasil disimpan di: {model_path}", flush=True)

if __name__ == '__main__':
    main()
