import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from preprocessing import DataPreprocessor
from model import build_multimodal_model
import os

# 1. FOOLPROOF PATH SETUP
# Get the folder where THIS script is located (e.g., .../Project/src)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go one level up to project root, then into 'output' (e.g., .../Project/output)
output_dir = os.path.join(os.path.dirname(script_dir), 'output')

# Ensure the folder exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"📂 Model will be saved to: {output_dir}")

def main():
    # 1. Prepare Data
    prep = DataPreprocessor()
    X_fused, y_fused = prep.fuse_datasets()
    
    # UNPACK 4 INPUTS
    X_clin, X_ctg, X_act, X_img = X_fused
    y_risk, y_weight = y_fused
    
    # 2. Split Data
    indices = np.arange(len(y_risk))
    # Stratify by risk to keep classes balanced
    X_clin_train, X_clin_test, y_risk_train, y_risk_test, idx_train, idx_test = train_test_split(
        X_clin, y_risk, indices, test_size=0.2, random_state=42, stratify=y_risk
    )
    
    # Split other modalities using indices
    X_ctg_train, X_ctg_test = X_ctg[idx_train], X_ctg[idx_test]
    X_act_train, X_act_test = X_act[idx_train], X_act[idx_test]
    X_img_train, X_img_test = X_img[idx_train], X_img[idx_test]
    
    y_weight_train, y_weight_test = y_weight[idx_train], y_weight[idx_test]
    
    # --- STRATEGY 1: COMPUTE SAMPLE WEIGHTS ---
    print("⚖️ Computing Weights to fix Imbalance...")
    unique_classes = np.unique(y_risk_train)
    
    # Calculate weight per class
    weights = class_weight.compute_class_weight(
        class_weight='balanced', 
        classes=unique_classes, 
        y=y_risk_train
    )
    class_weight_dict = dict(zip(unique_classes, weights))
    print(f"✅ Class Weights Map: {class_weight_dict}")

    # 1. Risk Weights: Map class ID to weight for every sample
    risk_sample_weights = np.array([class_weight_dict[y] for y in y_risk_train])
    
    # 2. Weight Regression Weights: Just use 1.0
    bw_sample_weights = np.ones((len(y_weight_train),))
    
    # 4. Build Quad-Modal Model
    input_shape_clin = (X_clin.shape[1],)
    input_shape_ctg = (X_ctg.shape[1], X_ctg.shape[2])
    input_shape_act = (X_act.shape[1], X_act.shape[2])
    input_shape_img = (128, 128, 1)
    
    model = build_multimodal_model(input_shape_clin, input_shape_ctg, input_shape_act, input_shape_img)
    
    # --- PATH FIX: Define the full path for the model file ---
    model_save_path = os.path.join(output_dir, 'best_maternal_model__v4.keras')

    # 5. Train
    callbacks = [
        # Use the absolute path variable 'model_save_path'
        ModelCheckpoint(model_save_path, monitor='val_output_risk_accuracy', save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    ]

    print(f"🚀 Starting Quad-Modal Training (Saving to {model_save_path})...")
    
    history = model.fit(
        x=[X_clin_train, X_ctg_train, X_act_train, X_img_train],
        y=[y_risk_train, y_weight_train],
        validation_data=(
            [X_clin_test, X_ctg_test, X_act_test, X_img_test], 
            [y_risk_test, y_weight_test]
        ),
        epochs=70,
        batch_size=32,
        sample_weight=[risk_sample_weights, bw_sample_weights],
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Evaluate
    print("\n📊 Evaluating...")
    # Use the absolute path variable again to load weights
    model.load_weights(model_save_path)
    
    results = model.evaluate(
        [X_clin_test, X_ctg_test, X_act_test, X_img_test],
        [y_risk_test, y_weight_test]
    )
    
    print(f"Final Metrics: {results}")

if __name__ == "__main__":
    main()