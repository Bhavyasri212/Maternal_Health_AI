import numpy as np
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocessing import DataPreprocessor
from model import build_multimodal_model
import os

# Create output folder
if not os.path.exists('../output'):
    os.makedirs('../output')

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
    model.load_weights('../output/best_maternal_model.keras')
    
    # 3. OPTIMIZATION: Prepare Static Modalities
    # We fix the CTG, Activity, and Image to their MEAN values.
    # We only want to explain the CLINICAL changes.
    mean_ctg = np.mean(X_ctg, axis=0, keepdims=True)
    mean_act = np.mean(X_act, axis=0, keepdims=True)
    mean_img = np.mean(X_img, axis=0, keepdims=True)
    
    def model_wrapper(clin_data):
        # Efficiently repeat the static data to match the batch size
        n = clin_data.shape[0]
        # We use 'broadcast_to' which is faster than repeat
        batch_ctg = np.tile(mean_ctg, (n, 1, 1))
        batch_act = np.tile(mean_act, (n, 1, 1))
        batch_img = np.tile(mean_img, (n, 1, 1, 1))
        
        # Predict Risk (Index 0)
        return model.predict([clin_data, batch_ctg, batch_act, batch_img], verbose=0)[0]

    # 4. OPTIMIZATION: K-Means Summary
    print("   ... Summarizing background data (Speed Boost 🚀)")
    # Instead of sending 1000 rows, we send 5 "representative" rows
    background_summary = shap.kmeans(X_clin, 5)
    
    explainer = shap.KernelExplainer(model_wrapper, background_summary)
    
    # 5. Calculate SHAP for just 5 samples (Enough for the report graph)
    print("   ... Calculating SHAP values (will finish quickly)")
    # 'nsamples=100' limits the precision slightly for massive speed
    test_samples = X_clin[100:105] 
    shap_values = explainer.shap_values(test_samples, nsamples=100)
    
    # 6. Handle Output
    if isinstance(shap_values, list):
        # Index 2 is High Risk
        vals_to_plot = shap_values[2] 
    else:
        vals_to_plot = shap_values[:, :, 2]

    # 7. Plot
    feature_names = [
        'Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate',
        'Sleep', 'Activity', 'Stress', 'Edu', 'Income', 'Urban',
        'DietQ', 'Hemo', 'Iron', 'Folic', 'DietAdh'
    ]
    # Clip names if mismatch
    feature_names = feature_names[:vals_to_plot.shape[1]]
    
    plt.figure()
    shap.summary_plot(vals_to_plot, test_samples, feature_names=feature_names, show=False)
    plt.title("Top Risk Factors (High Risk Class)")
    plt.savefig('../output/shap_summary_plot.png', bbox_inches='tight')
    print("✅ Saved SHAP Summary Plot to output/shap_summary_plot.png")

if __name__ == "__main__":
    try:
        explain_model()
    except Exception as e:
        print(f"⚠️ Critical SHAP Error: {e}")