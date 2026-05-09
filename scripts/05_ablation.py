#Current Main
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np
import tensorflow as tf
import random
import time
import itertools
from statsmodels.stats.contingency_tables import mcnemar

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt

# =====================================================================
# === Configuration ===
# =====================================================================
INPUT_FILE = 'final_dataset.csv'
SEED = 23
DNA_MAP = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}

# === OPTUNA BEST HYPERPARAMETERS FOR M1 & M3 ===
OPT_LR = 0.00037850170090655757
OPT_L2 = 0.00010103986497032854
OPT_ACTIVATION = 'elu'
OPT_SEQ_FILTERS = 128
OPT_ATAC_FILTERS = 32
OPT_FPN_F1 = 16
OPT_FPN_F2 = 32
OPT_FPN_DROP = 0.17253956663401096
OPT_DENSE_1 = 256
OPT_DENSE_2 = 32
OPT_DENSE_DROP_1 = 0.5894551302125344
OPT_DENSE_DROP_2 = 0.2952131315147431
OPT_BATCH_SIZE = 12

# =====================================================================
# DATA ENCODING & SPLITTING
# =====================================================================
def parse_dataset(df):
    print("Encoding data for Deep Learning (Only runs once!)...")
    n_samples = len(df)
    X_b1 = np.zeros((n_samples, 23, 8), dtype=np.float32)
    X_b2 = np.zeros((n_samples, 200, 5), dtype=np.float32)

    X_b3_1kb = np.zeros((n_samples, 11, 11, 1), dtype=np.float32)
    X_b3_10kb = np.zeros((n_samples, 11, 11, 1), dtype=np.float32)
    X_b3_50kb = np.zeros((n_samples, 11, 11, 1), dtype=np.float32)

    y = np.zeros(n_samples, dtype=np.float32)
    groups = np.zeros(n_samples, dtype=object)

    for i, row in enumerate(df.itertuples()):
        y[i] = float(row.label)
        groups[i] = str(row.grna_target_sequence)

        t_seq = str(row.target_sequence).upper().ljust(23, 'N')[:23]
        g_seq = str(row.grna_target_sequence).upper().ljust(23, 'N')[:23]
        for j in range(23):
            X_b1[i, j, :4] = DNA_MAP.get(t_seq[j], [0,0,0,0])
            X_b1[i, j, 4:] = DNA_MAP.get(g_seq[j], [0,0,0,0])

        c_seq = str(row.target_context).upper().ljust(200, 'N')[:200]
        try:
            atac_vals = [float(x) for x in str(row.atac_array).split(',')]
            if len(atac_vals) != 200: atac_vals = [0.0] * 200
        except:
            atac_vals = [0.0] * 200

        for j in range(200):
            X_b2[i, j, :4] = DNA_MAP.get(c_seq[j], [0,0,0,0])
            X_b2[i, j, 4] = atac_vals[j]

        try:
            hic_vals = [float(x) for x in str(row.hic_30x10_matrix).split(',')]
            if len(hic_vals) == 363:
                vals = np.array(hic_vals)
                X_b3_1kb[i, :, :, 0] = np.log1p(vals[0:121].reshape(11, 11))
                X_b3_10kb[i, :, :, 0] = np.log1p(vals[121:242].reshape(11, 11))
                X_b3_50kb[i, :, :, 0] = np.log1p(vals[242:363].reshape(11, 11))
        except:
            pass

    return X_b1, X_b2, X_b3_1kb, X_b3_10kb, X_b3_50kb, y, groups

# Custom Loss
def weighted_binary_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    pos = tf.reduce_sum(y_true)
    neg = tf.reduce_sum(1.0 - y_true)
    total = pos + neg

    pos_weight = total / (2.0 * tf.maximum(pos, epsilon))
    neg_weight = total / (2.0 * tf.maximum(neg, epsilon))

    loss = -(
        pos_weight * y_true * tf.math.log(y_pred) +
        neg_weight * (1.0 - y_true) * tf.math.log(1.0 - y_pred)
    )
    return tf.reduce_mean(loss)

# =====================================================================
# BASELINE BLOCKS (UNTOUCHED FOR M2, M4)
# =====================================================================
def get_b1(reg):
    b1_in = Input(shape=(23, 8), name="B1_Seq")
    x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=reg)(b1_in)
    x = BatchNormalization()(x)
    x = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    return b1_in, x

def get_b2(reg):
    b2_in = Input(shape=(200, 5), name="B2_Ctx_ATAC")
    x = Conv1D(64, 5, activation='relu', padding='same', kernel_regularizer=reg)(b2_in)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 5, activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    return b2_in, x

def build_dense_head(merged_tensor, reg):
    z = Dense(128, activation='relu', kernel_regularizer=reg)(merged_tensor)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    z = Dense(64, activation='relu', kernel_regularizer=reg)(z)
    z = Dropout(0.3)(z)
    out = Dense(1, activation='sigmoid', name="Prediction")(z)
    return out

# =====================================================================
# OPTIMIZED BLOCKS (STRICTLY FOR M1 & M3 FPN MODELS)
# =====================================================================
def build_model_1_concat_fusion_optimized():
    reg = tf.keras.regularizers.l2(OPT_L2)
    act = OPT_ACTIVATION

    # Opt Branch 1
    b1_in = Input(shape=(23, 8), name="B1_Seq_Opt")
    x1 = Conv1D(OPT_SEQ_FILTERS, 3, activation=act, padding='same', kernel_regularizer=reg)(b1_in)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(OPT_SEQ_FILTERS, 3, activation=act, padding='same', kernel_regularizer=reg)(x1)
    x1 = MaxPooling1D(2)(x1)
    x1 = Flatten()(x1)

    # Opt Branch 2
    b2_in = Input(shape=(200, 5), name="B2_Ctx_ATAC_Opt")
    x2 = Conv1D(OPT_ATAC_FILTERS, 5, activation=act, padding='same', kernel_regularizer=reg)(b2_in)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling1D(2)(x2)
    x2 = Conv1D(OPT_ATAC_FILTERS, 5, activation=act, padding='same', kernel_regularizer=reg)(x2)
    x2 = MaxPooling1D(2)(x2)
    x2 = Flatten()(x2)

    # Opt Branch 3 (FPN)
    in_1kb = Input(shape=(11, 11, 1), name="B3_HiC_1kb_Opt")
    in_10kb = Input(shape=(11, 11, 1), name="B3_HiC_10kb_Opt")
    in_50kb = Input(shape=(11, 11, 1), name="B3_HiC_50kb_Opt")

    def process_res_opt(res_in, name_prefix):
        x = Conv2D(OPT_FPN_F1, (3,3), activation=act, padding='same', kernel_regularizer=reg, name=f"{name_prefix}_conv1_opt")(res_in)
        x = BatchNormalization(name=f"{name_prefix}_bn1_opt")(x)
        x = MaxPooling2D((2,2), name=f"{name_prefix}_pool1_opt")(x)
        x = Dropout(OPT_FPN_DROP, name=f"{name_prefix}_drop1_opt")(x)

        x = Conv2D(OPT_FPN_F2, (3,3), activation=act, padding='same', kernel_regularizer=reg, name=f"{name_prefix}_conv2_opt")(x)
        x = BatchNormalization(name=f"{name_prefix}_bn2_opt")(x)
        x = MaxPooling2D((2,2), name=f"{name_prefix}_pool2_opt")(x)
        x = Dropout(OPT_FPN_DROP, name=f"{name_prefix}_drop2_opt")(x)
        return Flatten(name=f"{name_prefix}_flatten_opt")(x)

    out_1kb = process_res_opt(in_1kb, "1kb")
    out_10kb = process_res_opt(in_10kb, "10kb")
    out_50kb = process_res_opt(in_50kb, "50kb")

    merged_hic = Concatenate(name="FPN_Concat_Opt")([out_1kb, out_10kb, out_50kb])
    merged_hic = Dense(16, activation=act, kernel_regularizer=reg, name="FPN_Dense_Merge_Opt")(merged_hic)
    x3 = BatchNormalization(name="FPN_Merge_BN_Opt")(merged_hic)

    # Opt Fusion
    x1_rep = Dense(32, activation=act, kernel_regularizer=reg, name="Seq_Rep_Opt")(x1)
    x2_rep = Dense(32, activation=act, kernel_regularizer=reg, name="ATAC_Rep_Opt")(x2)
    x3_rep = Dense(32, activation=act, kernel_regularizer=reg, name="HiC_FPN_Rep_Opt")(x3)

    merged = Concatenate(name="Main_Fusion_Concat_Opt")([x1_rep, x2_rep, x3_rep])

    # Opt Dense Head
    z = Dense(OPT_DENSE_1, activation=act, kernel_regularizer=reg)(merged)
    z = BatchNormalization()(z)
    z = Dropout(OPT_DENSE_DROP_1)(z)
    z = Dense(OPT_DENSE_2, activation=act, kernel_regularizer=reg)(z)
    z = Dropout(OPT_DENSE_DROP_2)(z)
    out = Dense(1, activation='sigmoid', name="Prediction_Opt")(z)

    model_inputs = [b1_in, b2_in, in_1kb, in_10kb, in_50kb]
    model = Model(model_inputs, out, name="M1_Concat_FPN_Optimized")
    model.compile(optimizer=Adam(learning_rate=OPT_LR), loss=weighted_binary_crossentropy, metrics=[tf.keras.metrics.AUC(curve='PR', name='pr_auc')])
    return model

def build_model_3_seq_hic_optimized():
    reg = tf.keras.regularizers.l2(OPT_L2)
    act = OPT_ACTIVATION

    # Opt Branch 1
    b1_in = Input(shape=(23, 8), name="B1_Seq_Opt")
    x1 = Conv1D(OPT_SEQ_FILTERS, 3, activation=act, padding='same', kernel_regularizer=reg)(b1_in)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(OPT_SEQ_FILTERS, 3, activation=act, padding='same', kernel_regularizer=reg)(x1)
    x1 = MaxPooling1D(2)(x1)
    x1 = Flatten()(x1)

    # Opt Branch 3 (FPN)
    in_1kb = Input(shape=(11, 11, 1), name="B3_HiC_1kb_Opt")
    in_10kb = Input(shape=(11, 11, 1), name="B3_HiC_10kb_Opt")
    in_50kb = Input(shape=(11, 11, 1), name="B3_HiC_50kb_Opt")

    def process_res_opt(res_in, name_prefix):
        x = Conv2D(OPT_FPN_F1, (3,3), activation=act, padding='same', kernel_regularizer=reg, name=f"{name_prefix}_conv1_opt")(res_in)
        x = BatchNormalization(name=f"{name_prefix}_bn1_opt")(x)
        x = MaxPooling2D((2,2), name=f"{name_prefix}_pool1_opt")(x)
        x = Dropout(OPT_FPN_DROP, name=f"{name_prefix}_drop1_opt")(x)

        x = Conv2D(OPT_FPN_F2, (3,3), activation=act, padding='same', kernel_regularizer=reg, name=f"{name_prefix}_conv2_opt")(x)
        x = BatchNormalization(name=f"{name_prefix}_bn2_opt")(x)
        x = MaxPooling2D((2,2), name=f"{name_prefix}_pool2_opt")(x)
        x = Dropout(OPT_FPN_DROP, name=f"{name_prefix}_drop2_opt")(x)
        return Flatten(name=f"{name_prefix}_flatten_opt")(x)

    out_1kb = process_res_opt(in_1kb, "1kb")
    out_10kb = process_res_opt(in_10kb, "10kb")
    out_50kb = process_res_opt(in_50kb, "50kb")

    merged_hic = Concatenate(name="FPN_Concat_Opt")([out_1kb, out_10kb, out_50kb])
    merged_hic = Dense(16, activation=act, kernel_regularizer=reg, name="FPN_Dense_Merge_Opt")(merged_hic)
    x3 = BatchNormalization(name="FPN_Merge_BN_Opt")(merged_hic)

    # Opt Fusion (Only Seq + Hi-C)
    x1_rep = Dense(32, activation=act, kernel_regularizer=reg, name="Seq_Rep_Opt")(x1)
    x3_rep = Dense(32, activation=act, kernel_regularizer=reg, name="HiC_FPN_Rep_Opt")(x3)

    merged = Concatenate(name="Main_Fusion_Concat_Opt")([x1_rep, x3_rep])

    # Opt Dense Head
    z = Dense(OPT_DENSE_1, activation=act, kernel_regularizer=reg)(merged)
    z = BatchNormalization()(z)
    z = Dropout(OPT_DENSE_DROP_1)(z)
    z = Dense(OPT_DENSE_2, activation=act, kernel_regularizer=reg)(z)
    z = Dropout(OPT_DENSE_DROP_2)(z)
    out = Dense(1, activation='sigmoid', name="Prediction_Opt")(z)

    model_inputs = [b1_in, in_1kb, in_10kb, in_50kb]
    model = Model(model_inputs, out, name="M3_Seq_HiC_Optimized")
    model.compile(optimizer=Adam(learning_rate=OPT_LR), loss=weighted_binary_crossentropy, metrics=[tf.keras.metrics.AUC(curve='PR', name='pr_auc')])
    return model

# =====================================================================
# ABLATION MODELS (UNTOUCHED M2, M4)
# =====================================================================
def build_model_2_seq_ctx():
    reg = tf.keras.regularizers.l2(1e-4)
    b1_in, x1 = get_b1(reg)
    b2_in, x2 = get_b2(reg)

    x1_rep = Dense(32, activation='relu', kernel_regularizer=reg)(x1)
    x2_rep = Dense(32, activation='relu', kernel_regularizer=reg)(x2)

    merged = Concatenate()([x1_rep, x2_rep])
    out = build_dense_head(merged, reg)

    model = Model([b1_in, b2_in], out, name="M2_Seq_Ctx")
    model.compile(optimizer=Adam(learning_rate=0.001), loss=weighted_binary_crossentropy, metrics=[tf.keras.metrics.AUC(curve='PR', name='pr_auc')])
    return model


def build_model_4_seq_only():
    reg = tf.keras.regularizers.l2(1e-4)
    b1_in, x1 = get_b1(reg)

    x1_rep = Dense(32, activation='relu', kernel_regularizer=reg)(x1)
    out = build_dense_head(x1_rep, reg)

    model = Model(b1_in, out, name="M4_Seq_Only")
    model.compile(optimizer=Adam(learning_rate=0.001), loss=weighted_binary_crossentropy, metrics=[tf.keras.metrics.AUC(curve='PR', name='pr_auc')])
    return model

# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main():
    print("=== INITIALIZING DATASET (ONLY ONCE) ===")
    df = pd.read_csv(INPUT_FILE)
    X_b1, X_b2, X_b3_1kb, X_b3_10kb, X_b3_50kb, y, groups = parse_dataset(df)

    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)

    # Shuffle and Split based on the fixed seed
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(gss1.split(X_b1, y, groups=groups))

    temp_groups = groups[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_rel_idx, test_rel_idx = next(gss2.split(temp_idx, y[temp_idx], groups=temp_groups))
    val_idx, test_idx = temp_idx[val_rel_idx], temp_idx[test_rel_idx]

    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    class_weights = {0: 1.0, 1: min(15.0, (sum(y_train==0) / sum(y_train==1)) * 0.5)}

    def train_and_eval(model_func, train_inputs, val_inputs, test_inputs, save_name, batch_size=64):
        tf.keras.backend.clear_session()
        model = model_func()

        callbacks = [
            EarlyStopping(monitor='val_pr_auc', mode='max', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.5, patience=4),
            ModelCheckpoint(save_name, monitor='val_pr_auc', mode='max', save_best_only=True, verbose=0)
        ]

        print(f"Training {model.name} with batch size {batch_size}...")
        model.fit(
            train_inputs, y_train,
            validation_data=(val_inputs, y_val),
            epochs=45, batch_size=batch_size, class_weight=class_weights, callbacks=callbacks, verbose=0
        )

        model = tf.keras.models.load_model(save_name, compile=False)

        # Find the optimal threshold on the validation set right after training,
        # before testing begins.
        val_probs = model.predict(val_inputs, verbose=0).flatten()
        precision_v, recall_v, thresholds_v = precision_recall_curve(y_val, val_probs)
        f1_scores = np.divide(
            2 * (precision_v[:-1] * recall_v[:-1]),
            (precision_v[:-1] + recall_v[:-1]),
            out=np.zeros_like(precision_v[:-1]),
            where=(precision_v[:-1] + recall_v[:-1]) != 0
        )
        opt_idx = np.argmax(f1_scores)
        opt_thresh = thresholds_v[opt_idx]

        probs = model.predict(test_inputs, verbose=0).flatten()

        precision, recall, thresholds = precision_recall_curve(y_test, probs)

        binary_preds = (probs >= opt_thresh).astype(int)

        return {
            'probs': probs, 'preds': binary_preds,
            'pr_auc': average_precision_score(y_test, probs),
            'roc_auc': roc_auc_score(y_test, probs),
            'precision': precision, 'recall': recall,
            'fpr': roc_curve(y_test, probs)[0], 'tpr': roc_curve(y_test, probs)[1]
        }

    results = {}

    # --- M1 (Full Optimized) ---
    name_m1 = 'M1 (Full Concat Opt)'
    results[name_m1] = train_and_eval(
        build_model_1_concat_fusion_optimized,
        [X_b1[train_idx], X_b2[train_idx], X_b3_1kb[train_idx], X_b3_10kb[train_idx], X_b3_50kb[train_idx]],
        [X_b1[val_idx], X_b2[val_idx], X_b3_1kb[val_idx], X_b3_10kb[val_idx], X_b3_50kb[val_idx]],
        [X_b1[test_idx], X_b2[test_idx], X_b3_1kb[test_idx], X_b3_10kb[test_idx], X_b3_50kb[test_idx]],
        "m1_concat_fpn_opt.keras",
        batch_size=OPT_BATCH_SIZE
    )
    print(f"\n---> FINISHED {name_m1} <---")
    print(f"     PR-AUC: {results[name_m1]['pr_auc']:.4f}")
    print(f"     ROC-AUC: {results[name_m1]['roc_auc']:.4f}\n")

    # --- M2 (Seq + ATAC Baseline) ---
    name_m2 = 'M2 (Seq + ATAC)'
    results[name_m2] = train_and_eval(
        build_model_2_seq_ctx,
        [X_b1[train_idx], X_b2[train_idx]],
        [X_b1[val_idx], X_b2[val_idx]],
        [X_b1[test_idx], X_b2[test_idx]],
        "m2_seq_atac_baseline.keras"
    )
    print(f"\n---> FINISHED {name_m2} <---")
    print(f"     PR-AUC: {results[name_m2]['pr_auc']:.4f}")
    print(f"     ROC-AUC: {results[name_m2]['roc_auc']:.4f}\n")

    # --- M3 (Seq + HiC Optimized) ---
    name_m3 = 'M3 (Seq + HiC Opt)'
    results[name_m3] = train_and_eval(
        build_model_3_seq_hic_optimized,
        [X_b1[train_idx], X_b3_1kb[train_idx], X_b3_10kb[train_idx], X_b3_50kb[train_idx]],
        [X_b1[val_idx], X_b3_1kb[val_idx], X_b3_10kb[val_idx], X_b3_50kb[val_idx]],
        [X_b1[test_idx], X_b3_1kb[test_idx], X_b3_10kb[test_idx], X_b3_50kb[test_idx]],
        "m3_seq_hic_opt.keras",
        batch_size=OPT_BATCH_SIZE
    )
    print(f"\n---> FINISHED {name_m3} <---")
    print(f"     PR-AUC: {results[name_m3]['pr_auc']:.4f}")
    print(f"     ROC-AUC: {results[name_m3]['roc_auc']:.4f}\n")

    # --- M4 (Seq Only Baseline) ---
    name_m4 = 'M4 (Seq Only)'
    results[name_m4] = train_and_eval(
        build_model_4_seq_only,
        X_b1[train_idx],
        X_b1[val_idx],
        X_b1[test_idx],
        "m4_seq_only_baseline.keras"
    )
    print(f"\n---> FINISHED {name_m4} <---")
    print(f"     PR-AUC: {results[name_m4]['pr_auc']:.4f}")
    print(f"     ROC-AUC: {results[name_m4]['roc_auc']:.4f}\n")

    print("\n" + "="*50)
    print("=== ABLATION METRICS ===")
    for name, data in results.items():
        print(f"{name:20s} | PR-AUC: {data['pr_auc']:.4f} | ROC-AUC: {data['roc_auc']:.4f}")

    print("\n" + "="*50)
    print("=== MCNEMAR'S STATISTICAL SIGNIFICANCE TEST ===")
    print("="*50)

    models = list(results.keys())
    for (m_a, m_b) in itertools.combinations(models, 2):
        correct_a = (results[m_a]['preds'] == y_test)
        correct_b = (results[m_b]['preds'] == y_test)

        n00 = np.sum(correct_a & correct_b)
        n11 = np.sum(~correct_a & ~correct_b)
        n01 = np.sum(correct_a & ~correct_b)
        n10 = np.sum(~correct_a & correct_b)

        table = [[n00, n01], [n10, n11]]

        result = mcnemar(table, exact=False, correction=True)
        significance = "***" if result.pvalue < 0.001 else "**" if result.pvalue < 0.01 else "*" if result.pvalue < 0.05 else "n.s."
        print(f"{m_a} vs {m_b}:")
        print(f"  {m_a} correct / {m_b} wrong: {n01}")
        print(f"  {m_b} correct / {m_a} wrong: {n10}")
        print(f"  p-value: {result.pvalue:.4e} ({significance})\n")

    # =========================================================
    # === NEW CONFUSION MATRIX OUTPUT (M1 FULL MODEL) =========
    # =========================================================
    print("\n" + "="*50)
    print("=== CONFUSION MATRIX: M1 (Full Concat Opt) ===")
    print("="*50)
    m1_preds = results['M1 (Full Concat Opt)']['preds']
    cm = confusion_matrix(y_test, m1_preds)
    print("Test Set Confusion Matrix (Threshold optimized on Validation Set):")
    print(f"True Negatives (TN):  {cm[0, 0]}")
    print(f"False Positives (FP): {cm[0, 1]}")
    print(f"False Negatives (FN): {cm[1, 0]}")
    print(f"True Positives (TP):  {cm[1, 1]}")
    print("=================================================")

    # =========================================================
    # === NEW SEPARATED CURVE PLOTTING ========================
    # =========================================================
    print("\nGenerating Separated Overlaid Curves...")

    colors = {
        'M1 (Full Concat Opt)': '#E63946', 'M2 (Seq + ATAC)': '#F5A623',
        'M3 (Seq + HiC Opt)':  '#4A90E2', 'M4 (Seq Only)':   '#A0A0A0'
    }
    linestyles = {
        'M1 (Full Concat Opt)': '-', 'M2 (Seq + ATAC)': '--',
        'M3 (Seq + HiC Opt)':  '-.', 'M4 (Seq Only)':   ':'
    }

    # Custom mapping to standardize names cleanly on the final visual
    # without breaking any internal keys.
    display_names = {
        'M1 (Full Concat Opt)': 'TAG-Cas (Seq + ATAC + Hi-C)',
        'M2 (Seq + ATAC)': 'Baseline (Seq + ATAC)',
        'M3 (Seq + HiC Opt)':  'Baseline (Seq + Hi-C)',
        'M4 (Seq Only)':   'Baseline (Seq Only)'
    }

    # --- ROC Curve ---
    plt.figure(figsize=(8, 6))
    for name, data in results.items():
        disp_name = display_names[name]
        plt.plot(data['fpr'], data['tpr'], color=colors[name], linestyle=linestyles[name], lw=2.5, label=f"{disp_name} (AUC={data['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.title('Ablation Study: ROC Curves', fontweight='bold', fontsize=14)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('ablation_roc_curve_fpn_optimized.png', dpi=300)
    plt.close()

    # --- PR Curve ---
    plt.figure(figsize=(8, 6))
    for name, data in results.items():
        disp_name = display_names[name]
        plt.plot(data['recall'], data['precision'], color=colors[name], linestyle=linestyles[name], lw=2.5, label=f"{disp_name} (AUC={data['pr_auc']:.3f})")
    plt.title('Ablation Study: Precision-Recall Curves', fontweight='bold', fontsize=14)
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (Positive Predictive Value)')
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('ablation_pr_curve_fpn_optimized.png', dpi=300)
    plt.close()

    print(f"Saved separate visualizations to: ablation_roc_curve_fpn_optimized.png & ablation_pr_curve_fpn_optimized.png")
    print("=================================================")

if __name__ == "__main__":
    main()
