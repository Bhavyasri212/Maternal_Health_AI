import numpy as np
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocessing import DataPreprocessor
from model import build_multimodal_model
import os

# 1. FOOLPROOF PATH SETUP
# Get the folder where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go one level up to project root, then into 'output'
output_dir = os.path.join(os.path.dirname(script_dir), 'output')

# Ensure output folder exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"📂 SHAP Plots will be saved to: {output_dir}")

def explain_model():
    print("🧠 Initializing Fast SHAP (Optimized)...")
    
    # 1. Load Data
    prep = DataPreprocessor()
    X_fused, y_fused = prep.fuse_datasets()
    X_clin, X_ctg, X_act, X_img = X_fused
    
    # 2. Load Model
    print("   ... Loading model weights")
    model = build_multimodal_model(
        (X_clin.shape[1],), 
        (X_ctg.shape[1], X_ctg.shape[2]), 
        (X_act.shape[1], X_act.shape[2]),
        (128, 128, 1)
    )
    
    # PATH FIX: Load weights using absolute path
    model_path = os.path.join(output_dir, 'best_maternal_model__v4.keras')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at {model_path}. Please run train.py first.")
        
    model.load_weights(model_path)
    
    # 3. OPTIMIZATION: Prepare Static Modalities
    # We fix the CTG, Activity, and Image to their MEAN values.
    # We only want to explain the CLINICAL changes.
    mean_ctg = np.mean(X_ctg, axis=0, keepdims=True)
    mean_act = np.mean(X_act, axis=0, keepdims=True)
    mean_img = np.mean(X_img, axis=0, keepdims=True)
    
    def model_wrapper(clin_data):
        # Efficiently repeat the static data to match the batch size
        n = clin_data.shape[0]
        # We use 'tile' to broadcast the mean values
        batch_ctg = np.tile(mean_ctg, (n, 1, 1))
        batch_act = np.tile(mean_act, (n, 1, 1))
        batch_img = np.tile(mean_img, (n, 1, 1, 1))
        
        # Predict Risk (Index 0 is Risk Output)
        return model.predict([clin_data, batch_ctg, batch_act, batch_img], verbose=0)[0]

    # 4. OPTIMIZATION: K-Means Summary
    print("   ... Summarizing background data (Speed Boost 🚀)")
    # Instead of sending 1000 rows, we send 5 "representative" rows (centroids)
    background_summary = shap.kmeans(X_clin, 5)
    
    explainer = shap.KernelExplainer(model_wrapper, background_summary)
    
    # 5. Calculate SHAP for just 5 samples
    print("   ... Calculating SHAP values (will finish quickly)")
    # We take 5 real samples from the dataset to test
    test_samples = X_clin[100:105] 
    shap_values = explainer.shap_values(test_samples, nsamples=100)
    
    # 6. Handle Output Format (SHAP output format varies by version)
    vals_to_plot = None
    if isinstance(shap_values, list):
        # If list, usually [Low, Mid, High]. We want High Risk (Index 2)
        # Check list length to be safe
        target_idx = 2 if len(shap_values) > 2 else 0
        vals_to_plot = shap_values[target_idx] 
    else:
        # If array (Samples, Features, Classes)
        vals_to_plot = shap_values[:, :, 2]

    # 7. Plot
    feature_names = [
        'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate',
        'Sleep', 'Activity', 'Stress', 'Edu', 'Income', 'Urban',
        'DietQ', 'Hemo', 'Iron', 'Folic', 'DietAdh'
    ]
    # Clip names if mismatch in column count
    feature_names = feature_names[:vals_to_plot.shape[1]]
    
    plt.figure()
    shap.summary_plot(vals_to_plot, test_samples, feature_names=feature_names, show=False)
    plt.title("Top Risk Factors (High Risk Class)")
    
    # PATH FIX: Save to absolute path
    save_path = os.path.join(output_dir, 'shap_summary_plot.png')
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Saved SHAP Summary Plot to: {save_path}")

if __name__ == "__main__":
    try:
        explain_model()
    except Exception as e:
        print(f"⚠️ Critical SHAP Error: {e}")