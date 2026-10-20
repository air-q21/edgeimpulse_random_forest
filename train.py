import argparse
import os
import sys
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def main():
    # 1. Parsing argumen CLI & parameter dari Edge Impulse
    parser = argparse.ArgumentParser(description='Train Random Forest Classifier for Edge Impulse')
    parser.add_argument('--data-directory', type=str, required=True, help='Path direktori data input')
    parser.add_argument('--out-directory', type=str, required=True, help='Path direktori output model')
    parser.add_argument('--n_estimators', type=int, default=100, help='Jumlah pohon dalam Random Forest')
    parser.add_argument('--max_depth', type=int, default=10, help='Kedalaman maksimum pohon')
    
    # parse_known_args penting agar argumen bawaan Edge Impulse (--info-file, --epochs, dll) tidak bikin error
    args, _ = parser.parse_known_args()

    print(f"--> Memulai proses pelatihan dengan n_estimators={args.n_estimators}, max_depth={args.max_depth}")

    # 2. Lokasi file dataset
    x_train_path = os.path.join(args.data_directory, 'X_split_train.npy')
    y_train_path = os.path.join(args.data_directory, 'Y_split_train.npy')

    if not os.path.exists(x_train_path) or not os.path.exists(y_train_path):
        print(f"ERROR: File dataset tidak ditemukan di {args.data_directory}")
        sys.exit(1)

    # 3. Memuat data NumPy
    X_train = np.load(x_train_path)
    Y_train = np.load(y_train_path)

    print(f"--> Shape X_train awal: {X_train.shape}")
    print(f"--> Shape Y_train awal: {Y_train.shape}")

    # Flatten X_train jika fitur berbentuk multidimensi (misal dari Spectral Analysis / Image)
    if len(X_train.shape) > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)
        print(f"--> Reshaped X_train menjadi: {X_train.shape}")

    # 4. Mengambil label_index dari kolom pertama Y_train
    if Y_train.ndim > 1:
        y_labels = Y_train[:, 0].astype(int)
    else:
        y_labels = Y_train.astype(int)

    print(f"--> Total sampel: {X_train.shape[0]}, Total fitur per sampel: {X_train.shape[1]}")

    # 5. Pelatihan Model Random Forest
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1
    )

    print("--> Melatih Random Forest Classifier...")
    clf.fit(X_train, y_labels)
    print("--> Pelatihan selesai dengan sukses!")

    # 6. Menyimpan file model.pkl ke folder output wajib
    os.makedirs(args.out_directory, exist_ok=True)
    model_output_path = os.path.join(args.out_directory, 'model.pkl')
    
    joblib.dump(clf, model_output_path)
    print(f"--> Artefak model berhasil disimpan di: {model_output_path}")

if __name__ == '__main__':
    main()
