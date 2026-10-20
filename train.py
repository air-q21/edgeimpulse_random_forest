import argparse
import os
import sys
import traceback
import joblib
import numpy as np

# Pastikan pesan log langsung dicetak ke terminal Studio tanpa tertahan di buffer
def log(msg):
    print(f"[DEBUG] {msg}", flush=True)

def main():
    try:
        log("--> Memulai skrip pelatihan diagnostik...")
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--data-directory', type=str, required=True)
        parser.add_argument('--out-directory', type=str, required=True)
        parser.add_argument('--n_estimators', type=int, default=20)
        parser.add_argument('--max_depth', type=int, default=6)
        args, _ = parser.parse_known_args()

        log(f"Argumen diterima: data_dir={args.data_directory}, out_dir={args.out_directory}")
        log(f"Parameter model: n_estimators={args.n_estimators}, max_depth={args.max_depth}")

        # Cek versi library yang terpasang
        import sklearn
        log(f"Versi NumPy: {np.__version__}")
        log(f"Versi Scikit-Learn: {sklearn.__version__}")

        # Path file dataset
        x_path = os.path.join(args.data_directory, 'X_split_train.npy')
        y_path = os.path.join(args.data_directory, 'Y_split_train.npy')

        log(f"Mengecek keberadaan file dataset...")
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            log(f"ERROR: File {x_path} atau {y_path} tidak ditemukan!")
            sys.exit(1)

        # Memuat data
        log("Memuat dataset X_split_train.npy...")
        X_train = np.load(x_path).astype(np.float32)
        log(f"X_train berhasil dimuat. Shape: {X_train.shape}, Tipe: {X_train.dtype}")

        log("Memuat dataset Y_split_train.npy...")
        Y_train = np.load(y_path)
        log(f"Y_train berhasil dimuat. Shape: {Y_train.shape}, Tipe: {Y_train.dtype}")

        # Flatten X jika lebih dari 2D
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
            log(f"Reshaped X_train menjadi 2D: {X_train.shape}")

        # Penanganan Label Y
        if Y_train.ndim > 1:
            y_labels = Y_train[:, 0].astype(int)
        else:
            y_labels = Y_train.astype(int)
            
        log(f"Label Y siap. Jumlah kelas unik: {len(np.unique(y_labels))}")

        # Inisialisasi dan Pelatihan Model
        from sklearn.ensemble import RandomForestClassifier
        log("Menginisialisasi RandomForestClassifier...")
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=1
        )

        log("Memulai proses fitting model...")
        clf.fit(X_train, y_labels)
        log("Proses fitting berhasil!")

        # Menyimpan Model
        os.makedirs(args.out_directory, exist_ok=True)
        out_file = os.path.join(args.out_directory, 'model.pkl')
        log(f"Menyimpan model ke {out_file}...")
        joblib.dump(clf, out_file)
        log("Model model.pkl berhasil disimpan secara sempurna!")

    except Exception as e:
        log("=== TERJADI ERROR SAAT EKSEKUSI ===")
        log(f"Pesan Error: {str(e)}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)

if __name__ == '__main__':
    main()
