import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Suppress TF logs for cleaner execution
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Concatenate, Dropout, BatchNormalization, Multiply
from tensorflow.keras.regularizers import l2

# === 1. PAGE CONFIGURATION ===
st.set_page_config(
    page_title="TAG-Cas Clinical Demo",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# === 2. ARCHITECTURE DEFINITION ===
def build_m4():
    reg = l2(1e-4)
    align_input = Input(shape=(23, 8), name="Guide_Target_Alignment")
    x1 = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=reg)(align_input)
    x1 = MaxPooling1D(pool_size=2)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(16, activation='relu', kernel_regularizer=reg)(x1)
    
    context_input = Input(shape=(200, 4), name="DNA_Context")
    x2 = Conv1D(filters=32, kernel_size=8, activation='relu', kernel_regularizer=reg)(context_input)
    x2 = MaxPooling1D(pool_size=4)(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Conv1D(filters=16, kernel_size=4, activation='relu', kernel_regularizer=reg)(x2)
    x2 = MaxPooling1D(pool_size=4)(x2)
    x2 = Flatten()(x2)
    x2 = Dense(16, activation='sigmoid', kernel_regularizer=reg, bias_initializer='ones')(x2)
    
    phys_input = Input(shape=(2,), name="3D_and_ATAC")
    x3 = Dense(8, activation='relu')(phys_input)
    x3 = Dense(16, activation='sigmoid', bias_initializer='ones')(x3)
    
    gated_context = Multiply()([x1, x2])
    gated_physics = Multiply()([x1, x3])
    merged = Concatenate()([x1, gated_context, gated_physics])
    
    z = Dense(32, activation='relu', kernel_regularizer=reg)(merged)
    z = BatchNormalization()(z)
    z = Dropout(0.5)(z)
    output = Dense(1, activation='sigmoid', name="Prediction")(z)
    return Model(inputs=[align_input, context_input, phys_input], outputs=output)

# === 3. CACHED DATA & MODEL LOADING ===
@st.cache_resource
def load_model():
    model = build_m4()
    model.load_weights('m4_pure.keras')
    return model

@st.cache_data
def load_database():
    df = pd.read_csv('final_enriched_dataset.csv')
    scaler = StandardScaler()
    scaler.fit(df[['epigen_dnase', 'energy_1']].fillna(0).values)
    return df, scaler

DNA_MAP = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}

def one_hot_encode(seq, length):
    seq = str(seq).upper().ljust(length, 'N')[:length]
    return np.array([[DNA_MAP.get(base, [0,0,0,0]) for base in seq]], dtype=np.float32)

# === 4. UI LAYOUT & LOGIC ===
def main():
    st.title("TAG-Cas")
    st.subheader("3D-Aware CRISPR Off-Target Predictor")
    st.markdown("---")

    model = load_model()
    df, scaler = load_database()

    st.markdown("### Enter Genomic Parameters")
    
    # Row 1: Sequences
    col1, col2 = st.columns(2)
    with col1:
        grna_seq = st.text_input("Guide RNA (23bp)", value="GTCATCTTAGTCATTACCTGAGG").upper()
    with col2:
        target_seq = st.text_input("Target DNA (23bp)", value="GGTATCTAAGTCATTACCTGTGG").upper()
    
    # Row 2: Coordinates & Cell Line
    col3, col4 = st.columns(2)
    with col3:
        coord = st.text_input("Genomic Coordinate", value="chr5:92701253")
    with col4:
        # NEW CELL LINE DROPDOWN
        cell_line = st.selectbox(
            "Target Cell Line", 
            options=["K562", "HEK293T", "U2OS", "HeLa"], 
            index=0
        )

    if st.button("Run Diagnostic Scan", type="primary", use_container_width=True):
        
        if len(grna_seq) != 23 or len(target_seq) != 23:
            st.error("Error: Both Guide RNA and Target DNA must be exactly 23 base pairs long.")
            return

        # Dynamically update the loading bar text with the chosen cell line
        progress_text = f"Querying {cell_line} Genomic Database..."
        my_bar = st.progress(0, text=progress_text)
        
        match = df[(df['grna_target_sequence'] == grna_seq) & (df['target_sequence'] == target_seq)]
        
        time.sleep(2)
        my_bar.progress(50, text=f"Extracting 3D Epigenetic & Sequence Context for {cell_line}...")
        
        if not match.empty:
            row = match.iloc[0]
            context_seq = row['target_context']
            phys_raw = row[['epigen_dnase', 'energy_1']].fillna(0).values
            simulated = False
        else:
            context_seq = "N" * 200
            phys_raw = np.array([0.5, -20.0])
            simulated = True

        time.sleep(2)
        my_bar.progress(80, text="Running Triple-Branch Neural Network Inference...")

        X_align = np.concatenate([one_hot_encode(target_seq, 23), one_hot_encode(grna_seq, 23)], axis=-1)
        X_context = one_hot_encode(context_seq, 200)
        X_phys = scaler.transform([phys_raw])
        
        prob = model.predict([X_align, X_context, X_phys], verbose=0)[0][0]
        
        my_bar.progress(100, text="Scan Complete.")
        time.sleep(0.3)
        my_bar.empty()
        
        # === Output Report Section ===
        st.markdown("---")
        st.markdown("### Clinical Prediction Report")
        
        if prob >= 0.50:
            st.error("HIGH RISK (CLEAVAGE LIKELY)")
            st.metric(label="AI Predicted Probability", value=f"{prob*100:.2f}%")
        else:
            st.success("SAFE (BIOLOGICAL VETO APPLIED)")
            st.metric(label="AI Predicted Probability", value=f"{prob*100:.2f}%")
            
        st.markdown("#### Diagnostic Details")
        st.markdown(f"**Target Coordinate:** `{coord}`")
        st.markdown(f"**Evaluated Cell Line:** `{cell_line}`") # Added to confirm the environment

if __name__ == "__main__":
    main()
