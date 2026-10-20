import argparse
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def main():
    parser = argparse.ArgumentParser(description='Train Random Forest Classifier for Edge Impulse')
    parser.add_argument('--data-directory', type=str, required=True)
    parser.add_argument('--out-directory', type=str, required=True)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    args, _ = parser.parse_known_args()

    # Memuat dataset fitur dan label dari Edge Impulse
    X_train = np.load(os.path.join(args.data_directory, 'X_split_train.npy'))
    Y_train = np.load(os.path.join(args.data_directory, 'Y_split_train.npy'))

    # Mengambil index label kelas dari kolom pertama
    if Y_train.ndim > 1 and Y_train.shape[1] > 0:
        y_labels = Y_train[:, 0]
    else:
        y_labels = Y_train.ravel()

    # Pelatihan model Random Forest Classifier
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    clf.fit(X_train, y_labels)

    # Menyimpan artefak model.pkl ke folder output
    os.makedirs(args.out_directory, exist_ok=True)
    model_path = os.path.join(args.out_directory, 'model.pkl')
    joblib.dump(clf, model_path)
    print(f"Model berhasil disimpan di: {model_path}")

if __name__ == '__main__':
    main()