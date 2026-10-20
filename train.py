import argparse
import os
import sys
import traceback
import joblib
import numpy as np

def log(msg):
    print(f"[EI-LOG] {msg}", flush=True)

def main():
    try:
        log("--> Memulai skrip pelatihan Random Forest...")

        parser = argparse.ArgumentParser()
        parser.add_argument('--data-directory', type=str, required=True)
        parser.add_argument('--out-directory', type=str, required=True)
        parser.add_argument('--n_estimators', type=int, default=25)
        parser.add_argument('--max_depth', type=int, default=8)
        args, _ = parser.parse_known_args()

        log(f"Parameter: n_estimators={args.n_estimators}, max_depth={args.max_depth}")

        # Jalur data
        x_path = os.path.join(args.data_directory, 'X_split_train.npy')
        y_path = os.path.join(args.data_directory, 'Y_split_train.npy')

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            log(f"ERROR: Dataset tidak ditemukan di {args.data_directory}")
            sys.exit(1)

        # Memuat data & optimasi RAM (float32)
        log("Memuat X_split_train.npy...")
        X_train = np.load(x_path).astype(np.float32)
        log(f"X_train shape: {X_train.shape}")

        log("Memuat Y_split_train.npy...")
        Y_train = np.load(y_path)

        # Reshape jika data fitur berbentuk >2D
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
            log(f"Reshaped X_train ke 2D: {X_train.shape}")

        # Ambil kolom label
        if Y_train.ndim > 1:
            y_labels = Y_train[:, 0].astype(int)
        else:
            y_labels = Y_train.astype(int)

        # Training Random Forest
        from sklearn.ensemble import RandomForestClassifier
        log("Membuat instance RandomForestClassifier...")
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            max_features='sqrt',
            random_state=42,
            n_jobs=1
        )

        log("Memulai fitting data...")
        clf.fit(X_train, y_labels)
        log("Fitting selesai!")

        # Simpan model.pkl
        os.makedirs(args.out_directory, exist_ok=True)
        out_file = os.path.join(args.out_directory, 'model.pkl')
        joblib.dump(clf, out_file)
        log(f"Model berhasil disimpan di: {out_file}")

    except Exception as e:
        log("=== ERROR EKSEKUSI ===")
        log(str(e))
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)

if __name__ == '__main__':
    main()
