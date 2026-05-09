import os
import time
import csv
from pathlib import Path

import cooler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyBigWig
from Bio import SeqIO
from matplotlib.patches import Rectangle

# === Configuration ===
input_file = "final_training_dataset.csv"
output_file = "out.csv"

# Data Sources
genome_path = "hg38.fa"
atac_path = "K562_ATAC_Accessibility.bigWig"
hic_path = "K562_HiC_3D_Contacts.mcool"

# Parameters
CONTEXT_WINDOW = 100        # +/- 100bp around target (Total 200bp context)
HIC_RESOLUTIONS = [1000, 10000, 50000] # Local, Mid, High Range
HIC_MATRIX_BINS = 11        # 11x11 matrix per resolution
NUM_EXAMPLES = 10           # Hard stop after this many rows
VISUAL_DIR = "science_fair_visuals_new_arch"


def chromosome_sort_key(chrom):
    """Sort chr1..chr22, chrX, chrY, chrM in natural genomic order."""
    if pd.isna(chrom):
        return 10_000
    chrom = str(chrom).replace("chr", "")
    if chrom.isdigit():
        return int(chrom)
    special = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    return special.get(chrom.upper(), 10_000)


def ensure_visual_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_available_hic_resolutions(mcool_path):
    try:
        names = cooler.fileops.list_coolers(mcool_path)
        resolutions = []
        for name in names:
            if "/resolutions/" in name:
                tail = name.split("/resolutions/")[-1]
                if tail.isdigit():
                    resolutions.append(int(tail))
        return sorted(set(resolutions))
    except Exception as e:
        print(f"WARNING: Could not list cooler resolutions: {e}")
        return []


def open_hic_cooler(mcool_path, requested_resolution):
    available = get_available_hic_resolutions(mcool_path)
    chosen_resolution = requested_resolution
    if available and requested_resolution not in available:
        chosen_resolution = min(available, key=lambda x: abs(x - requested_resolution))
        print(f"WARNING: Requested resolution {requested_resolution}bp not available. Using {chosen_resolution}bp.")

    try:
        c = cooler.Cooler(f"{mcool_path}::/resolutions/{chosen_resolution}")
    except Exception as e:
        print(f"FATAL ERROR: Could not load Hi-C resolution {chosen_resolution}bp. {e}")
        return None, None, available, chosen_resolution

    actual_binsize = getattr(c, "binsize", chosen_resolution)
    return c, actual_binsize, available, chosen_resolution


def fetch_hic_window(c, chrom, center_bp, matrix_bins):
    binsize = int(getattr(c, "binsize", 1000))
    half = matrix_bins // 2

    try:
        chrom_bins = c.bins().fetch(chrom)
        if len(chrom_bins) == 0:
            raise ValueError(f"No bins found for {chrom}")

        starts = chrom_bins["start"].to_numpy()
        ends = chrom_bins["end"].to_numpy()

        idx_local = int(np.searchsorted(starts, center_bp, side="right") - 1)
        if idx_local < 0 or idx_local >= len(chrom_bins):
            raise ValueError(f"Center {center_bp} outside bins")
        if not (starts[idx_local] <= center_bp < ends[idx_local]):
            raise ValueError(f"Center {center_bp} not inside valid bin")

        center_bin_global = int(chrom_bins.index[idx_local])
    except Exception as e:
        M = np.zeros((matrix_bins, matrix_bins), dtype=np.float32)
        return M, {"status": f"error: {e}", "center_bin": -1}

    chrom_start_bin, chrom_end_bin = c.extent(chrom)
    desired_start_bin = center_bin_global - half
    desired_end_bin = center_bin_global + half + 1

    start_bin = max(chrom_start_bin, desired_start_bin)
    end_bin = min(chrom_end_bin, desired_end_bin)

    M = np.zeros((matrix_bins, matrix_bins), dtype=np.float32)
    status = "ok"

    try:
        if start_bin < end_bin:
            sub = c.matrix(balance=False, sparse=False)[start_bin:end_bin, start_bin:end_bin]
            sub = np.asarray(sub, dtype=np.float32)

            row_offset = start_bin - desired_start_bin
            col_offset = start_bin - desired_start_bin

            r1 = min(matrix_bins, row_offset + sub.shape[0])
            c1 = min(matrix_bins, col_offset + sub.shape[1])

            M[row_offset:r1, col_offset:c1] = sub[:(r1 - row_offset), :(c1 - col_offset)]
    except Exception as e:
        status = f"fetch_error: {e}"

    return M, {"status": status, "center_bin": center_bin_global}


# --- Visuals ---
def save_atac_visual(example, path):
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(example["ctx_start"], example["ctx_end"])
    y = example["atac_array_vals"]

    ax.plot(x, y, color="#0B7285", lw=2.0)
    ax.fill_between(x, y, 0, color="#74C0FC", alpha=0.45)
    ax.axvline(example["center"], color="#C2255C", lw=2.5, linestyle="--")

    ax.set_title(f"Branch 2: 200bp Epigenetic Footprint Tensor", fontsize=18, fontweight="bold")
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis("off")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_hic_visual(matrix, resolution, path):
    fig = plt.figure(figsize=(8, 8))
    ax0 = fig.add_subplot(111)

    if matrix is None or matrix.sum() == 0:
        ax0.axis("off")
        ax0.text(0.05, 0.5, f"Sparse Matrix (No significant 3D contacts)\nat {resolution}bp resolution.", fontsize=14, color="#7F1D1D", ha='left')
    else:
        img = ax0.imshow(np.log1p(matrix), cmap="magma", interpolation="nearest", aspect="auto")
        center_idx = matrix.shape[0] // 2 
        ax0.plot(center_idx, center_idx, marker="x", color="#80ED99", markersize=10, mew=2)
        ax0.add_patch(Rectangle((center_idx - 0.5, center_idx - 0.5), 1, 1, fill=False, lw=2, ec="#B2F2BB"))
        ax0.set_title(f"FPN Topology Layer: {resolution}bp Map", fontsize=16, fontweight="bold")
        cb = fig.colorbar(img, ax=ax0, fraction=0.046, pad=0.04)
        cb.set_label("log1p(contact count)")

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def extract_features_optimized_with_visuals():
    print("--- Starting Phase 2: Visual Generation Fast-Track ---")
    start_time = time.time()
    ensure_visual_dir(VISUAL_DIR)

    df = pd.read_csv(input_file)
    df["sort_key"] = df["target_chr"].apply(chromosome_sort_key)
    df = df.sort_values(by=["sort_key", "target_start"]).drop(columns=["sort_key"]).reset_index(drop=True)
    
    genome_idx = SeqIO.index(genome_path, "fasta")

    print("Connecting to Hi-C Pyramids...")
    coolers_info = []
    for res in HIC_RESOLUTIONS:
        c, actual_res, _, _ = open_hic_cooler(hic_path, res)
        if c is None: return
        coolers_info.append({"c": c, "actual": actual_res})

    bw = pyBigWig.open(atac_path)
    example_records = []

    # Setup CSV writing
    drop_cols = ["epigen_dnase", "energy_1", "epigen_ctcf", "epigen_rrbs", "epigen_h3k4me3", "epigen_drip", "hic_21x21_matrix"]
    out_columns = [col for col in df.columns if col not in drop_cols]
    new_csv_headers = ["target_context", "target_strand", "atac_array", "hic_30x10_matrix", "hic_actual_resolutions", "hic_center_bins", "hic_nonzero_bins", "hic_sum_contacts", "hic_status"]
    out_columns.extend(new_csv_headers)
    
    out_f = open(output_file, 'w', newline='')
    csv_writer = csv.DictWriter(out_f, fieldnames=out_columns)
    csv_writer.writeheader()

    print(f"Extracting first {NUM_EXAMPLES} rows for visuals...")

    for i, row in enumerate(df.itertuples(index=False), start=1):
        chrom = row.target_chr
        start = int(row.target_start)
        end = int(row.target_end)
        center = (start + end) // 2
        ctx_start = max(0, center - CONTEXT_WINDOW)
        ctx_end = center + CONTEXT_WINDOW

        # Sequence
        try:
            raw_seq = genome_idx[chrom][ctx_start:ctx_end].seq
            context_seq = str(raw_seq).upper()
            strand = "+" if str(row.target_strand) == "0" else str(row.target_strand)
        except KeyError:
            context_seq = "N" * (CONTEXT_WINDOW * 2)
            strand = "+"

        # ATAC
        try:
            atac_vals = bw.values(chrom, ctx_start, ctx_end, numpy=True)
            atac_vals = np.nan_to_num(atac_vals, nan=0.0)
            if len(atac_vals) != (CONTEXT_WINDOW * 2): atac_vals = np.zeros(CONTEXT_WINDOW * 2)
        except:
            atac_vals = np.zeros(CONTEXT_WINDOW * 2)

        # Hi-C
        M_blocks, meta_blocks = [], []
        for info in coolers_info:
            M_sub, meta_sub = fetch_hic_window(info["c"], chrom, center, HIC_MATRIX_BINS)
            M_blocks.append(M_sub)
            meta_blocks.append(meta_sub)
            
        M_stacked = np.vstack(M_blocks)
        
        # Write to CSV
        row_dict = {col: getattr(row, col) for col in df.columns}
        for drop in drop_cols: row_dict.pop(drop, None)
        row_dict.update({
            "target_context": context_seq, "target_strand": strand,
            "atac_array": ",".join([f"{x:.4f}" for x in atac_vals]),
            "hic_30x10_matrix": ",".join([f"{x:.1f}" for x in M_stacked.flatten()]),
            "hic_actual_resolutions": "|".join([str(info["actual"]) for info in coolers_info]),
            "hic_center_bins": "|".join([str(m["center_bin"]) for m in meta_blocks]),
            "hic_nonzero_bins": int(np.count_nonzero(M_stacked)),
            "hic_sum_contacts": float(M_stacked.sum()),
            "hic_status": "|".join([m["status"] for m in meta_blocks])
        })
        csv_writer.writerow(row_dict)

        # Save for plotting
        example_records.append({
            "example_id": i, "chrom": chrom, "center": center, "ctx_start": ctx_start, "ctx_end": ctx_end,
            "atac_array_vals": atac_vals, "hic_matrices": M_blocks, "resolutions": HIC_RESOLUTIONS
        })

        # === THE HARD STOP ===
        if i >= NUM_EXAMPLES:
            print(f"\n[INFO] Reached {NUM_EXAMPLES} examples. Halting extraction to generate visuals immediately.")
            break

    out_f.close()
    bw.close()

    print("\nGenerating SPARTAN visual proofs for FPN...")
    for ex in example_records:
        idx = ex["example_id"]
        save_atac_visual(ex, os.path.join(VISUAL_DIR, f"example_{idx:02d}_atac_tensor.png"))
        for j, res in enumerate(ex["resolutions"]):
            save_hic_visual(ex["hic_matrices"][j], res, os.path.join(VISUAL_DIR, f"example_{idx:02d}_hic_{res}bp.png"))

    print(f"Finished in {time.time() - start_time:.1f} seconds. Visuals saved to '{VISUAL_DIR}/'")

if __name__ == "__main__":
    extract_features_optimized_with_visuals()
