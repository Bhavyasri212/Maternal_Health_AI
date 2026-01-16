import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from preprocessing import DataPreprocessor
from model import build_multimodal_model

# 1. FOOLPROOF PATH SETUP
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), 'output')
model_path = os.path.join(output_dir, 'best_maternal_model__v4.keras')

def run_final_test():
    print("\n" + "="*50)
    print("🧪 INITIALIZING FINAL TEST MODE")
    print("="*50)

    # 1. Load All Data
    print("📂 Loading Dataset...")
    prep = DataPreprocessor()
    X_fused, y_fused = prep.fuse_datasets()
    
    X_clin, X_ctg, X_act, X_img = X_fused
    y_risk, y_weight = y_fused

    # 2. Re-create the EXACT same Split
    # We use random_state=42 to guarantee this 'test set' is the 
    # exact same 20% chunk the model was tested on before.
    indices = np.arange(len(y_risk))
    _, X_clin_test, _, y_risk_test, idx_train, idx_test = train_test_split(
        X_clin, y_risk, indices, test_size=0.2, random_state=42, stratify=y_risk
    )
    
    # Select the Test portions for other modalities
    X_ctg_test = X_ctg[idx_test]
    X_act_test = X_act[idx_test]
    X_img_test = X_img[idx_test]
    y_weight_test = y_weight[idx_test]
    
    print(f"✅ Test Set Loaded: {len(X_clin_test)} patients (Unseen Data)")

    # 3. Load the Saved Brain
    print(f"🧠 Loading Model from: {model_path}")
    if not os.path.exists(model_path):
        print("❌ CRITICAL ERROR: Model file not found!")
        return

    # We rebuild the structure so we can load the weights into it
    model = build_multimodal_model(
        (X_clin.shape[1],), 
        (X_ctg.shape[1], X_ctg.shape[2]), 
        (X_act.shape[1], X_act.shape[2]), 
        (128, 128, 1)
    )
    model.load_weights(model_path)
    
    # 4. Compile (Required for evaluate)
    model.compile(
        loss={'output_risk': 'sparse_categorical_crossentropy', 'output_weight': 'mae'},
        metrics={'output_risk': 'accuracy', 'output_weight': 'mae'}
    )

    # 5. Run the "Board Exam"
    print("\n🚀 Running Final Evaluation on Test Set...")
    results = model.evaluate(
        [X_clin_test, X_ctg_test, X_act_test, X_img_test],
        [y_risk_test, y_weight_test],
        verbose=1
    )

    # 6. Print Report Card
    print("\n" + "="*50)
    print("🏆 FINAL TEST RESULTS")
    print("="*50)
    print(f"✅ TEST ACCURACY (Risk):   {results[3]*100:.2f}%")
    print(f"✅ TEST ERROR (Weight):    {results[4]:.2f} grams")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_final_test()