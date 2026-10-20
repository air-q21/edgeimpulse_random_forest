import argparse
import os
import sys
import joblib
import numpy as np

print("=== CONTAINER ONLINE & PYTHON RUNNING ===", flush=True)

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-directory', type=str, required=True)
    parser.add_argument('--out-directory', type=str, required=True)
    parser.add_argument('--n_estimators', type=int, default=25)
    parser.add_argument('--max_depth', type=int, default=8)
    
    args, _ = parser.parse_known_args()

    print(f"Reading data from: {args.data_directory}", flush=True)

    X_path = os.path.join(args.data_directory, 'X_split_train.npy')
    Y_path = os.path.join(args.data_directory, 'Y_split_train.npy')

    X_train = np.load(X_path).astype(np.float32)
    Y_train = np.load(Y_path)

    if len(X_train.shape) > 2:
        X_train = X_train.reshape(X_train.shape[0], -1)

    y_labels = Y_train[:, 0].astype(int) if Y_train.ndim > 1 else Y_train.astype(int)

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators, 
        max_depth=args.max_depth, 
        random_state=42, 
        n_jobs=1
    )
    
    print("Training Random Forest Classifier...", flush=True)
    clf.fit(X_train, y_labels)

    os.makedirs(args.out_directory, exist_ok=True)
    model_path = os.path.join(args.out_directory, 'model.pkl')
    joblib.dump(clf, model_path)

    print("=== MODEL SAVED SUCCESSFULLY ===", flush=True)

except Exception as e:
    print(f"PYTHON ERROR: {str(e)}", flush=True)
    sys.exit(1)
    
