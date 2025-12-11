# ml_core.py
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from typing import Tuple, Dict, Any, List

# sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    xgb = None
    HAS_XGB = False

# tensorflow / keras (optional - heavy). If unavailable, deep models are skipped.
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.utils import plot_model
    HAS_TF = True
except Exception:
    tf = None
    layers = None
    models = None
    optimizers = None
    callbacks = None
    plot_model = None
    HAS_TF = False

# TCN (optional)
try:
    from tcn import TCN
    HAS_TCN = True
except Exception:
    TCN = None
    HAS_TCN = False

# visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# helper
import joblib

# -------------------------
# Utility / Preprocessing
# -------------------------
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def detect_feature_types(df: pd.DataFrame, target_col: str):
    features = [c for c in df.columns if c != target_col]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical

def preprocess_tabular(df: pd.DataFrame, target_col: str):
    # Basic: impute numeric with median, categorical with most_frequent; encode categorical with OneHot
    numeric, categorical = detect_feature_types(df, target_col)
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Simple label encoding for y if non-numeric
    if y.dtype == 'O' or not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        le = None

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ]) if len(categorical) > 0 else None

    if categorical_transformer:
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric),
            ('cat', categorical_transformer, categorical)
        ])
    else:
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric)
        ])
    X_processed = preprocessor.fit_transform(X)
    feature_names = []
    # try to extract feature names
    try:
        num_names = numeric
        cat_names = list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical)) if categorical_transformer else []
        feature_names = num_names + cat_names
    except Exception:
        feature_names = [f"f{i}" for i in range(X_processed.shape[1])]
    return X_processed, y, preprocessor, le, feature_names

def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = 10):
    """
    Create sliding windows for sequential models.
    X shape: (n_samples, n_features) -> output: (n_sequences, seq_len, n_features)
    y is aligned so that each sequence has a label of the last item in the window.
    """
    n_samples, n_features = X.shape
    sequences = []
    seq_labels = []
    for i in range(n_samples - seq_len + 1):
        sequences.append(X[i:i+seq_len])
        seq_labels.append(y[i+seq_len-1])
    Xs = np.stack(sequences)
    ys = np.array(seq_labels)
    return Xs, ys

# -------------------------
# Traditional Model Training
# -------------------------
def train_traditional_models(X, y, cv_splits=5):
    results = {}
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    # include XGBoost only if available
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'NaiveBayes': GaussianNB(),
        'RandomForest': RandomForestClassifier(n_estimators=200),
        'SVM': SVC(probability=True)
    }
    if HAS_XGB:
        try:
            models['XGBoost'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        except Exception:
            pass
    metrics_accum = {name: defaultdict(list) for name in models.keys()}

    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        for name, mdl in models.items():
            mdl.fit(Xtr, ytr)
            preds = mdl.predict(Xte)
            probs = None
            if hasattr(mdl, "predict_proba"):
                probs = mdl.predict_proba(Xte)[:, 1] if probs is None and mdl.predict_proba(Xte).shape[1] > 1 else mdl.predict_proba(Xte)[:, 0]
            elif hasattr(mdl, "decision_function"):
                # approximate probabilities with decision_function via sigmoid is not done here; fallback
                probs = mdl.decision_function(Xte)
            acc = accuracy_score(yte, preds)
            f1 = f1_score(yte, preds, average='weighted')
            auc_val = None
            try:
                # try to compute ROC AUC for binary or multilabel averaged
                if probs is not None and len(np.unique(y)) == 2:
                    auc_val = roc_auc_score(yte, probs)
                elif probs is not None and probs.ndim == 2:
                    # multiclass: one-vs-rest average
                    auc_val = roc_auc_score(yte, mdl.predict_proba(Xte), multi_class='ovr')
            except Exception:
                auc_val = None

            metrics_accum[name]['accuracy'].append(acc)
            metrics_accum[name]['f1'].append(f1)
            metrics_accum[name]['auc'].append(auc_val)

    # average metrics
    for name in models.keys():
        results[name] = {
            'accuracy_mean': float(np.mean(metrics_accum[name]['accuracy'])),
            'accuracy_std': float(np.std(metrics_accum[name]['accuracy'])),
            'f1_mean': float(np.mean(metrics_accum[name]['f1'])),
            'f1_std': float(np.std(metrics_accum[name]['f1'])),
            'auc_mean': float(np.nanmean([v for v in metrics_accum[name]['auc'] if v is not None])) if any(v is not None for v in metrics_accum[name]['auc']) else None,
            'auc_std': float(np.nanstd([v for v in metrics_accum[name]['auc'] if v is not None])) if any(v is not None for v in metrics_accum[name]['auc']) else None
        }
    return results, models

# -------------------------
# Deep Model builders
# -------------------------
def build_lstm(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax' if num_classes>2 else 'sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy' if num_classes>2 else 'binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_rnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.SimpleRNN(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_classes, activation='softmax' if num_classes>2 else 'sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy' if num_classes>2 else 'binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_cnn_1d(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax' if num_classes>2 else 'sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy' if num_classes>2 else 'binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_tcn(input_shape, num_classes):
    if not HAS_TCN:
        raise ImportError('TCN package not available in this environment')
    i = layers.Input(shape=input_shape)
    x = TCN(nb_filters=64, kernel_size=3, dilations=[1,2,4], return_sequences=False)(i)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax' if num_classes>2 else 'sigmoid')(x)
    model = models.Model(inputs=i, outputs=out)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy' if num_classes>2 else 'binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_lstm_rnn_hybrid(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.SimpleRNN(32)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax' if num_classes>2 else 'sigmoid')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy' if num_classes>2 else 'binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_cnn_lstm_hybrid(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.TimeDistributed(layers.Conv1D(32,3, activation='relu'))(inp) if len(input_shape)>1 else layers.Conv1D(32,3, activation='relu')(inp)
    x = layers.TimeDistributed(layers.MaxPooling1D(2))(x) if len(input_shape)>1 else layers.MaxPooling1D(2)(x)
    x = layers.TimeDistributed(layers.Flatten())(x) if len(input_shape)>1 else layers.Flatten()(x)
    x = layers.LSTM(64)(x) if len(input_shape)>1 else layers.Reshape((1, -1))(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax' if num_classes>2 else 'sigmoid')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy' if num_classes>2 else 'binary_crossentropy',
                  metrics=['accuracy'])
    return model

# -------------------------
# Deep training (K-Fold)
# -------------------------
if not HAS_TF:
    def train_deep_models(X, y, sequence_mode=False, seq_len=10, folds=3, epochs=20, batch_size=32, output_dir='static/plots'):
        """
        TensorFlow not available in environment — skip deep model training.
        """
        return {}
else:
    def train_deep_models(X, y, sequence_mode=False, seq_len=10, folds=3, epochs=20, batch_size=32, output_dir='static/plots'):
        """
        X: tabular processed np array (n_samples, n_features)
        If sequence_mode: create sequences and train LSTM/RNN/TCN etc on shape (n_seq, seq_len, n_features)
        Returns: dict of metrics and optionally saved architecture images
        """
        results = {}
        if sequence_mode:
            Xs, ys = create_sequences(X, y, seq_len)
        else:
            # for non sequence: convert to (n_samples, timesteps, features) with timesteps=1
            Xs = X.reshape((X.shape[0], 1, X.shape[1]))
            ys = y

        num_classes = len(np.unique(ys))
        # model builders map
        builders = {
            'LSTM': build_lstm,
            'RNN': build_rnn,
            'CNN_1D': build_cnn_1d,
            'TCN': build_tcn,
            'LSTM_RNN_Hybrid': build_lstm_rnn_hybrid,
            'CNN_LSTM_Hybrid': build_cnn_lstm_hybrid
        }

        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        for name, builder in builders.items():
            accs = []
            f1s = []
            aucs = []
            for train_idx, test_idx in kf.split(Xs):
                Xtr, Xte = Xs[train_idx], Xs[test_idx]
                ytr, yte = ys[train_idx], ys[test_idx]

                model = builder(input_shape=Xtr.shape[1:], num_classes=num_classes)
                es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                history = model.fit(Xtr, ytr, validation_split=0.15, epochs=epochs, batch_size=batch_size, callbacks=[es], verbose=0)

                preds = model.predict(Xte)
                if preds.ndim > 1 and preds.shape[1] > 1:
                    pred_labels = np.argmax(preds, axis=1)
                    probs = preds[:,1] if preds.shape[1] > 1 else preds[:,0]
                else:
                    pred_labels = (preds.ravel() > 0.5).astype(int)
                    probs = preds.ravel()

                accs.append(accuracy_score(yte, pred_labels))
                f1s.append(f1_score(yte, pred_labels, average='weighted'))
                try:
                    if num_classes == 2:
                        aucs.append(roc_auc_score(yte, probs))
                    else:
                        aucs.append(roc_auc_score(yte, preds, multi_class='ovr'))
                except Exception:
                    aucs.append(None)

            results[name] = {
                'accuracy_mean': float(np.mean(accs)),
                'accuracy_std': float(np.std(accs)),
                'f1_mean': float(np.mean(f1s)),
                'f1_std': float(np.std(f1s)),
                'auc_mean': float(np.nanmean([a for a in aucs if a is not None])) if any(a is not None for a in aucs) else None,
                'auc_std': float(np.nanstd([a for a in aucs if a is not None])) if any(a is not None for a in aucs) else None
            }

            # Save a placeholder architecture diagram (use plot_model if available)
            arch_path = os.path.join(output_dir, f'{name}_architecture.png')
            try:
                plot_model(builder(input_shape=Xs.shape[1:], num_classes=num_classes), to_file=arch_path, show_shapes=True)
                results[name]['architecture'] = arch_path
            except Exception:
                # fallback: save a simple text image describing the architecture
                plt.figure(figsize=(6,2))
                plt.text(0.01, 0.5, f"{name} architecture (see docstrings)", fontsize=10)
                plt.axis('off')
                plt.savefig(arch_path, bbox_inches='tight')
                plt.close()
                results[name]['architecture'] = arch_path

        return results

# -------------------------
# Visualization helpers
# -------------------------
def plot_roc_curve_multimodel(models_dict, X_test, y_test, preprocessor, output_path, is_deep=False, sequence_mode=False, seq_len=10):
    """
    models_dict: for traditional: fitted models; for deep we may pass a dict of (name->(model, predict_func))
    But in our case we will generate coarse ROC based on model predictions stored or re-run models on a held-out set.
    For simplicity, this function expects a dict of name->(predict_proba_func)
    """
    plt.figure(figsize=(8,6))
    for name, predict_proba_func in models_dict.items():
        try:
            probs = predict_proba_func(X_test)
            if probs.ndim > 1 and probs.shape[1] > 1:
                # take second class if binary; or compute multiclass OVR AUC
                if probs.shape[1] == 2:
                    probs_pos = probs[:,1]
                    fpr, tpr, _ = roc_curve(y_test, probs_pos)
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
                else:
                    # approximate by averaging one-vs-rest curves (not plotted here)
                    pass
            else:
                fpr, tpr, _ = roc_curve(y_test, probs)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.3f})')
        except Exception as e:
            # skip if cannot compute
            continue
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix_heatmap(cm, labels, output_path):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_comparative_bar(metric_dict, metric_key, output_path):
    # metric_dict: {model_name: metric_value}
    names = list(metric_dict.keys())
    vals = [metric_dict[n] for n in names]
    plt.figure(figsize=(10,6))
    sns.barplot(x=vals, y=names)
    plt.xlabel(metric_key)
    plt.title(f'Comparative {metric_key} across models')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# -------------------------
# High-level pipeline
# -------------------------
def run_full_pipeline(csv_path: str, target_col='target', sequence_mode=False, sequence_length=10, output_dir='static/plots'):
    os.makedirs(output_dir, exist_ok=True)
    df = load_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not found in CSV. Available columns: {list(df.columns)}")
    # preprocess
    X_proc, y_proc, preprocessor, label_encoder, feature_names = preprocess_tabular(df, target_col)
    # create hold-out test set for final visualizations & model-level function evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y_proc, test_size=0.15, stratify=y_proc, random_state=42)

    # Train traditional models (cross-validated)
    trad_results, trained_models = train_traditional_models(np.vstack([X_train, X_test]), np.hstack([y_train, y_test]), cv_splits=5)
    # NOTE: trained_models here are classifiers from last fold - we will re-fit them on full train for prediction uses below
    # Re-fit them on entire train (train+val subset, excluding holdout test) for producing ROC/CM on test
    retrained_for_plots = {}
    base_models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'NaiveBayes': GaussianNB(),
        'RandomForest': RandomForestClassifier(n_estimators=200),
        'SVM': SVC(probability=True)
    }
    if HAS_XGB:
        try:
            base_models['XGBoost'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        except Exception:
            pass

    for name, mdl in base_models.items():
        try:
            mdl.fit(X_train, y_train)
            retrained_for_plots[name] = mdl
        except Exception:
            # skip models that fail to fit in this environment
            continue

    # Train deep models (K-Fold) if TensorFlow is available
    if HAS_TF:
        deep_results = train_deep_models(np.vstack([X_train, X_test]), np.hstack([y_train, y_test]),
                                        sequence_mode=sequence_mode, seq_len=sequence_length, folds=3, epochs=20, output_dir=output_dir)
    else:
        deep_results = {}

    # Prepare aggregated metrics
    metrics_table = {}
    for name, v in trad_results.items():
        metrics_table[name] = {
            'accuracy': v['accuracy_mean'],
            'f1': v['f1_mean'],
            'auc': v['auc_mean']
        }
    for name, v in deep_results.items():
        metrics_table[name] = {
            'accuracy': v['accuracy_mean'],
            'f1': v['f1_mean'],
            'auc': v['auc_mean']
        }

    # Visualizations
    # 1) Comparative bar chart (accuracy)
    bar_path = os.path.join(output_dir, 'comparative_accuracy.png')
    plot_comparative_bar({k: metrics_table[k]['accuracy'] for k in metrics_table}, 'Accuracy', bar_path)

    # 2) Confusion matrix - pick best model by accuracy for demo
    best_model_name = max(metrics_table.items(), key=lambda x: x[1]['accuracy'])[0]
    # use retrained_for_plots if traditional else cannot easily produce predictions; for simplicity, if best is deep, we skip CM compute or use last deep model predictions (not stored) => produce placeholder
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    if best_model_name in retrained_for_plots:
        mdl = retrained_for_plots[best_model_name]
        preds = mdl.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        labels = [str(i) for i in np.unique(y_test)]
        plot_confusion_matrix_heatmap(cm, labels, cm_path)
    else:
        # produce placeholder image noting we don't have a single saved deep-model instance for direct predictions
        plt.figure(figsize=(6,2))
        plt.text(0.01, 0.5, f"No direct deep model instance saved for '{best_model_name}'.\nConfusion matrix unavailable.", fontsize=10)
        plt.axis('off')
        plt.savefig(cm_path)
        plt.close()

    # 3) ROC Curve - build simple dict of predict_proba functions for each traditional model
    roc_funcs = {}
    for name, mdl in retrained_for_plots.items():
        def make_fn(m):
            return lambda X_in: m.predict_proba(X_in) if hasattr(m, 'predict_proba') else np.vstack([1 - m.decision_function(X_in), m.decision_function(X_in)]).T
        roc_funcs[name] = make_fn(mdl)
    # for deep models we skip embedding predict_proba wrappers (would need re-fitted model objects). Keep ROC for traditional ones.
    roc_path = os.path.join(output_dir, 'roc_comparison.png')
    try:
        plot_roc_curve_multimodel(roc_funcs, X_test, y_test, preprocessor, roc_path)
    except Exception as e:
        # fallback placeholder
        plt.figure(figsize=(6,2))
        plt.text(0.01, 0.5, "ROC generation failed: "+str(e), fontsize=10)
        plt.axis('off')
        plt.savefig(roc_path)
        plt.close()

    # model architecture placeholders (deep_results already includes paths when built)
    architectures = {name: deep_results[name].get('architecture') for name in deep_results}

    # Package final response
    response = {
        'summary': {
            'num_samples': int(df.shape[0]),
            'num_features': int(X_proc.shape[1]),
            'models_trained': list(metrics_table.keys())
        },
        'metrics_table': metrics_table,
        'plots': {
            'comparative_accuracy': bar_path,
            'roc_comparison': roc_path,
            'confusion_matrix': cm_path
        },
        'architectures': architectures
    }

    return response
