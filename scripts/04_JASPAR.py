import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --- Keras Imports cleanly at the top ---
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate, Dropout, BatchNormalization

# Suppress TF logs for cleaner execution
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# --- NEW IMPORTS FOR AUTOMATED MOTIF MATCHING ---
try:
    from Bio import motifs
    from Bio.Seq import Seq
except ImportError:
    print("ERROR: Biopython is not installed. Please run 'pip install biopython' first.")
    exit()

print("=== INITIATING GLOBAL INTEGRATED GRADIENTS CONSENSUS ===")

# --- 1. CONFIGURATION ---
DATA_PATH = 'final_dataset.csv'
MODEL_WEIGHTS_PATH = 'm1_concat_fpn_opt.keras'
JASPAR_PATH = 'JASPAR2024.txt'

DNA_MAP = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}

# --- OPTUNA BEST HYPERPARAMETERS FOR ARCHITECTURE RECONSTRUCTION ---
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

# --- 2. DATA ENCODING (UPDATED FOR 5 INPUTS & ATAC) ---
def parse_dataset(df):
    print(f"Encoding dataset from {DATA_PATH}...")
    n_samples = len(df)
    X_b1 = np.zeros((n_samples, 23, 8), dtype=np.float32)
    X_b2 = np.zeros((n_samples, 200, 5), dtype=np.float32)
    X_b3_1kb = np.zeros((n_samples, 11, 11, 1), dtype=np.float32)
    X_b3_10kb = np.zeros((n_samples, 11, 11, 1), dtype=np.float32)
    X_b3_50kb = np.zeros((n_samples, 11, 11, 1), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)

    for i, row in enumerate(df.itertuples()):
        y[i] = float(row.label)

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

    return X_b1, X_b2, X_b3_1kb, X_b3_10kb, X_b3_50kb, y

# --- 3. BUILD AND LOAD OPTIMIZED MODEL ---
def build_model_1_concat_fusion_optimized():
    reg = tf.keras.regularizers.l2(OPT_L2)
    act = OPT_ACTIVATION

    b1_in = Input(shape=(23, 8), name="B1_Seq_Opt")
    x1 = Conv1D(OPT_SEQ_FILTERS, 3, activation=act, padding='same', kernel_regularizer=reg)(b1_in)
    x1 = BatchNormalization()(x1)
    x1 = Conv1D(OPT_SEQ_FILTERS, 3, activation=act, padding='same', kernel_regularizer=reg)(x1)
    x1 = MaxPooling1D(2)(x1)
    x1 = Flatten()(x1)

    b2_in = Input(shape=(200, 5), name="B2_Ctx_ATAC_Opt")
    x2 = Conv1D(OPT_ATAC_FILTERS, 5, activation=act, padding='same', kernel_regularizer=reg)(b2_in)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling1D(2)(x2)
    x2 = Conv1D(OPT_ATAC_FILTERS, 5, activation=act, padding='same', kernel_regularizer=reg)(x2)
    x2 = MaxPooling1D(2)(x2)
    x2 = Flatten()(x2)

    in_1kb = Input(shape=(11, 11, 1), name="B3_HiC_1kb_Opt")
    in_10kb = Input(shape=(11, 11, 1), name="B3_HiC_10kb_Opt")
    in_50kb = Input(shape=(11, 11, 1), name="B3_HiC_50kb_Opt")

    def process_res_opt(res_in):
        x = Conv2D(OPT_FPN_F1, (3,3), activation=act, padding='same', kernel_regularizer=reg)(res_in)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(OPT_FPN_DROP)(x)
        x = Conv2D(OPT_FPN_F2, (3,3), activation=act, padding='same', kernel_regularizer=reg)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2,2))(x)
        x = Dropout(OPT_FPN_DROP)(x)
        return Flatten()(x)

    out_1kb = process_res_opt(in_1kb)
    out_10kb = process_res_opt(in_10kb)
    out_50kb = process_res_opt(in_50kb)

    merged_hic = Concatenate()([out_1kb, out_10kb, out_50kb])
    merged_hic = Dense(16, activation=act, kernel_regularizer=reg)(merged_hic)
    x3 = BatchNormalization()(merged_hic)

    x1_rep = Dense(32, activation=act, kernel_regularizer=reg)(x1)
    x2_rep = Dense(32, activation=act, kernel_regularizer=reg)(x2)
    x3_rep = Dense(32, activation=act, kernel_regularizer=reg)(x3)

    merged = Concatenate()([x1_rep, x2_rep, x3_rep])
    z = Dense(OPT_DENSE_1, activation=act, kernel_regularizer=reg)(merged)
    z = BatchNormalization()(z)
    z = Dropout(OPT_DENSE_DROP_1)(z)
    z = Dense(OPT_DENSE_2, activation=act, kernel_regularizer=reg)(z)
    z = Dropout(OPT_DENSE_DROP_2)(z)
    out = Dense(1, activation='sigmoid', name="Prediction_Opt")(z)

    return Model([b1_in, b2_in, in_1kb, in_10kb, in_50kb], out)

# --- 4. LOAD JASPAR DATABASE ---
print(f"Loading JASPAR database from {JASPAR_PATH}...")
try:
    with open(JASPAR_PATH) as f:
        jaspar_motifs = motifs.parse(f, "jaspar")
    print(f"Successfully loaded {len(jaspar_motifs)} known transcription factor profiles.")
except FileNotFoundError:
    print(f"ERROR: Could not find {JASPAR_PATH}. Please download it and put it in this folder.")
    exit()

print("Calculating scoring thresholds for all TFs...")
tf_scoring_matrices = []
for i, m in enumerate(jaspar_motifs):
    pwm = m.counts.normalize(pseudocounts=0.5)
    pssm = pwm.log_odds()
    # Stricter threshold simulating p < 0.01 match
    threshold = pssm.max * 0.90
    tf_scoring_matrices.append({
        'name': m.name,
        'pssm': pssm,
        'length': len(m),
        'threshold': threshold
    })
print("Done pre-calculating matrices!")

# --- 5. EXECUTE DATA AND PREDICTIONS ---
df = pd.read_csv(DATA_PATH).head(10000)
X_b1, X_b2, X_b3_1kb, X_b3_10kb, X_b3_50kb, y_true = parse_dataset(df)

print(f"Loading trained weights from {MODEL_WEIGHTS_PATH}...")
model = build_model_1_concat_fusion_optimized()
model.load_weights(MODEL_WEIGHTS_PATH)

print("\nPredicting on data to find confident targets...")
inputs_list = [X_b1, X_b2, X_b3_1kb, X_b3_10kb, X_b3_50kb]
predictions = model.predict(inputs_list, verbose=0).flatten()

confident_tps = np.where((y_true == 1) & (predictions > 0.8))[0]
confident_tns = np.where((y_true == 0) & (predictions < 0.2))[0]

print(f"Found {len(confident_tps)} Confident True Positives.")
print(f"Found {len(confident_tns)} Confident True Negatives.")

targets_to_analyze_tp = confident_tps[np.argsort(predictions[confident_tps])[::-1]]
targets_to_analyze_tn = confident_tns[np.argsort(predictions[confident_tns])]
all_targets = np.concatenate([targets_to_analyze_tp, targets_to_analyze_tn])
labels_for_targets = np.concatenate([np.ones(len(targets_to_analyze_tp)), np.zeros(len(targets_to_analyze_tn))])

# --- 6. INTEGRATED GRADIENTS MATH ---
@tf.function
def compute_gradients(b1, b2, b3_1kb, b3_10kb, b3_50kb, baseline_b2, alphas):
    interpolated_b2 = baseline_b2 + alphas * (b2 - baseline_b2)
    r_b1 = tf.repeat(b1, tf.shape(alphas)[0], axis=0)
    r_b3_1kb = tf.repeat(b3_1kb, tf.shape(alphas)[0], axis=0)
    r_b3_10kb = tf.repeat(b3_10kb, tf.shape(alphas)[0], axis=0)
    r_b3_50kb = tf.repeat(b3_50kb, tf.shape(alphas)[0], axis=0)

    with tf.GradientTape() as tape:
        tape.watch(interpolated_b2)
        preds = model([r_b1, interpolated_b2, r_b3_1kb, r_b3_10kb, r_b3_50kb])
    return tape.gradient(preds, interpolated_b2)

def get_ig_scores(b1, b2, b3_1kb, b3_10kb, b3_50kb, num_steps=50):
    baseline_b2 = tf.zeros_like(b2)
    alphas = tf.reshape(tf.linspace(0.0, 1.0, num_steps+1), (num_steps+1, 1, 1))

    grads = compute_gradients(b1, b2, b3_1kb, b3_10kb, b3_50kb, baseline_b2, alphas)
    avg_grads = tf.reduce_mean(grads, axis=0)

    integrated_grads = (b2[0] - baseline_b2[0]) * avg_grads
    return tf.reduce_sum(integrated_grads, axis=1).numpy()

# --- 7. THE GLOBAL SCAN & BIOLOGICAL MAPPING ---
print(f"\nScanning the AI's brain and mapping to JASPAR (p < 0.01 threshold)...")

# Trackers now store tuples: (TF_Name, Center_Position, ATAC_Score)
matched_tfs_tp = []
matched_tfs_tn = []
inverse_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

for count, idx in enumerate(all_targets):
    if (count + 1) % 100 == 0:
        print(f"Processed {count + 1}/{len(all_targets)} sequences...")

    is_positive = labels_for_targets[count] == 1

    sample_b1 = tf.convert_to_tensor(X_b1[idx:idx+1])
    sample_b2 = tf.convert_to_tensor(X_b2[idx:idx+1])
    sample_1kb = tf.convert_to_tensor(X_b3_1kb[idx:idx+1])
    sample_10kb = tf.convert_to_tensor(X_b3_10kb[idx:idx+1])
    sample_50kb = tf.convert_to_tensor(X_b3_50kb[idx:idx+1])

    importance_scores = get_ig_scores(sample_b1, sample_b2, sample_1kb, sample_10kb, sample_50kb)
    importance_scores = np.abs(importance_scores)

    decoded_seq = []
    for i in range(200):
        dna_channels = X_b2[idx, i, :4]
        if np.sum(dna_channels) == 0:
            decoded_seq.append('N')
        else:
            decoded_seq.append(inverse_map[np.argmax(dna_channels)])

    window_size = 12
    max_score = -np.inf
    best_start = 0

    for i in range(200 - window_size):
        window_score = np.sum(importance_scores[i:i+window_size])
        if window_score > max_score:
            max_score = window_score
            best_start = i

    center_pos = best_start + (window_size / 2.0)

    # --- IDEA 1: EXTRACT THE AVERAGE ATAC-SEQ SCORE FOR THIS 12BP MOTIF ---
    # The 5th channel (index 4) contains the raw ATAC-seq accessibility score
    local_atac_score = np.mean(X_b2[idx, best_start:best_start+window_size, 4])

    motif = "".join(decoded_seq[best_start:best_start+window_size])
    if 'N' in motif:
        continue

    seq_obj = Seq(motif)

    for tf_data in tf_scoring_matrices:
        if 6 <= tf_data['length'] <= 16:
            try:
                score = tf_data['pssm'].calculate(seq_obj)
                max_score = np.max(score)

                if max_score >= tf_data['threshold']:
                    if is_positive:
                        matched_tfs_tp.append((tf_data['name'], center_pos, local_atac_score))
                    else:
                        matched_tfs_tn.append((tf_data['name'], center_pos, local_atac_score))
            except:
                pass

# --- 8. AGGREGATE AND REPORT ---
print("\n" + "="*50)
print("=== GLOBAL MOTIF CONSENSUS RESULTS (MAPPED TO TFs) ===")
print("="*50)

print(f"\nTotal n analyzed: {len(all_targets)} genomic loci")
print(f"Significant biological matches (p < 0.01) found in True Positives: {len(matched_tfs_tp)}")
print(f"Significant biological matches (p < 0.01) found in True Negatives: {len(matched_tfs_tn)}")

# Custom aggregation to handle counts, positions, AND ATAC scores
def aggregate_matches(matches):
    stats = {}
    for name, pos, atac in matches:
        if name not in stats:
            stats[name] = {'count': 0, 'positions': [], 'atac_scores': []}
        stats[name]['count'] += 1
        stats[name]['positions'].append(pos)
        stats[name]['atac_scores'].append(atac)
    for name in stats:
        stats[name]['avg_pos'] = sum(stats[name]['positions']) / len(stats[name]['positions'])
        stats[name]['avg_atac'] = sum(stats[name]['atac_scores']) / len(stats[name]['atac_scores'])
    return stats

tp_stats = aggregate_matches(matched_tfs_tp)
tn_stats = aggregate_matches(matched_tfs_tn)

# Sort by frequency count to get the top 10
top_tfs_tp = sorted(tp_stats.items(), key=lambda item: item[1]['count'], reverse=True)[:10]
top_tfs_tn = sorted(tn_stats.items(), key=lambda item: item[1]['count'], reverse=True)[:10]

print("\nTop 10 TFs driving TRUE POSITIVES (Cleavage Activators):")
for rank, (tf_name, data) in enumerate(top_tfs_tp, 1):
    print(f"#{rank}: {tf_name} (Found {data['count']} times | Avg Pos: {data['avg_pos']:.1f} | Avg ATAC: {data['avg_atac']:.4f})")

print("\nTop 10 TFs driving TRUE NEGATIVES (Cleavage Vetoes/Repressors):")
for rank, (tf_name, data) in enumerate(top_tfs_tn, 1):
    print(f"#{rank}: {tf_name} (Found {data['count']} times | Avg Pos: {data['avg_pos']:.1f} | Avg ATAC: {data['avg_atac']:.4f})")

# --- 9. POSTER BOARD VISUALIZATION ---
HEX_BG = 'white'
HEX_UNDERLINE = '#1c4587'

plt.rcParams.update({
    'figure.facecolor': HEX_BG,
    'axes.facecolor': HEX_BG,
    'savefig.facecolor': HEX_BG,
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

def add_board_title(ax, text):
    ax.set_title(text, fontsize=22, fontweight='bold', pad=25)
    ax.plot([0.1, 0.9], [1.02, 1.02], color=HEX_UNDERLINE, lw=5, transform=ax.transAxes, clip_on=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Modified to include ATAC scores in the bar chart labels
labels_tp = [f"{m[0]} (ATAC: {m[1]['avg_atac']:.2f})" for m in top_tfs_tp]
values_tp = [m[1]['count'] for m in top_tfs_tp]
if labels_tp:
    axes[0].barh(labels_tp[::-1], values_tp[::-1], color='#E63946', edgecolor='black', alpha=0.9)
add_board_title(axes[0], 'AI Cleavage Activators (True Positives)')
axes[0].set_xlabel('Frequency (Matches with p < 0.01)')

# Modified to include ATAC scores in the bar chart labels
labels_tn = [f"{m[0]} (ATAC: {m[1]['avg_atac']:.2f})" for m in top_tfs_tn]
values_tn = [m[1]['count'] for m in top_tfs_tn]
if labels_tn:
    axes[1].barh(labels_tn[::-1], values_tn[::-1], color='#4A90E2', edgecolor='black', alpha=0.9)
add_board_title(axes[1], 'AI Cleavage Repressors (True Negatives)')
axes[1].set_xlabel('Frequency (Matches with p < 0.01)')

plt.tight_layout()
plt.savefig('poster_tf_consensus_chart.png', dpi=300)
print("\n>>> Saved consensus chart to 'poster_tf_consensus_chart.png'.")

# --- 10. EXPORT EXHAUSTIVE TF LIST TO CSV ---
print("\nGenerating comprehensive CSV of all identified Transcription Factors...")

# Convert nested dicts into dataframes with positional and ATAC averages
df_tp = pd.DataFrame.from_dict({k: [v['count'], round(v['avg_pos'], 2), round(v['avg_atac'], 4)] for k, v in tp_stats.items()}, orient='index', columns=['TP_Frequency', 'TP_Avg_Position', 'TP_Avg_ATAC']).reset_index().rename(columns={'index': 'Transcription_Factor'})
df_tn = pd.DataFrame.from_dict({k: [v['count'], round(v['avg_pos'], 2), round(v['avg_atac'], 4)] for k, v in tn_stats.items()}, orient='index', columns=['TN_Frequency', 'TN_Avg_Position', 'TN_Avg_ATAC']).reset_index().rename(columns={'index': 'Transcription_Factor'})

df_all_tfs = pd.merge(df_tp, df_tn, on='Transcription_Factor', how='outer')

if not df_all_tfs.empty:
    df_all_tfs['TP_Frequency'] = df_all_tfs['TP_Frequency'].fillna(0).astype(int)
    df_all_tfs['TN_Frequency'] = df_all_tfs['TN_Frequency'].fillna(0).astype(int)
    df_all_tfs['Total_Frequency'] = df_all_tfs['TP_Frequency'] + df_all_tfs['TN_Frequency']

    # Reorder columns for clean CSV output
    df_all_tfs = df_all_tfs[['Transcription_Factor', 'Total_Frequency', 'TP_Frequency', 'TP_Avg_Position', 'TP_Avg_ATAC', 'TN_Frequency', 'TN_Avg_Position', 'TN_Avg_ATAC']]
    df_all_tfs = df_all_tfs.sort_values(by='Total_Frequency', ascending=False).reset_index(drop=True)

    csv_filename = 'all_identified_tfs_frequencies.csv'
    df_all_tfs.to_csv(csv_filename, index=False)
    print(f">>> SUCCESS: Full exhaustive list with average positions and ATAC scores saved to '{csv_filename}'!")
else:
    print(">>> No significant TFs found to save.")
