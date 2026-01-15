import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from preprocessing import DataPreprocessor
from model import build_multimodal_model
import os

# 1. DIAGNOSTIC PATH SETUP
# This gets the folder where THIS FILE (evaluate_proof.py) lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# We assume 'output' is one level up from 'src'
output_dir = os.path.join(os.path.dirname(script_dir), 'output')
model_path = os.path.join(output_dir, 'best_maternal_model__v4.keras')

print("\n" + "="*60)
print("🔍 DIAGNOSTIC PATH CHECK")
print("="*60)
print(f"📂 Script Location:   {script_dir}")
print(f"📂 Output Folder:     {output_dir}")
print(f"🧠 Looking for Model: {model_path}")

# Check if output directory exists
if not os.path.exists(output_dir):
    print("❌ ERROR: Output folder does not exist!")
    # Attempt to create it just in case, though it won't have the model
    os.makedirs(output_dir)
else:
    print("✅ Output folder exists.")
    # List files to see if the model is actually there
    files = os.listdir(output_dir)
    print(f"📂 Files found in output: {files}")

# Check if model exists
if not os.path.exists(model_path):
    print("\n❌ CRITICAL ERROR: 'best_maternal_model.keras' NOT FOUND.")
    print("   Please run 'python src/train.py' first to create the model.")
    exit() # Stop the script here
else:
    print("✅ Model file found! Proceeding with evaluation...")
print("="*60 + "\n")

def plot_confusion_matrix(y_true, y_pred_classes):
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Mid', 'High'], 
                yticklabels=['Low', 'Mid', 'High'])
    plt.xlabel('Predicted Risk')
    plt.ylabel('Actual Risk')
    plt.title('Confusion Matrix: Maternal Risk')
    
    save_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved Confusion Matrix -> {save_path}")

def plot_weight_scatter(y_true, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='purple')
    
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    plt.xlabel('Actual Birth Weight (g)')
    plt.ylabel('Predicted Birth Weight (g)')
    plt.title('Regression Analysis: Actual vs Predicted Weight')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    save_path = os.path.join(output_dir, 'weight_scatter_plot.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved Weight Scatter Plot -> {save_path}")

def plot_multiclass_roc(y_true, y_pred_probs):
    n_classes = 3
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=n_classes)
    class_names = ['Low Risk', 'Mid Risk', 'High Risk']
    colors = ['green', 'orange', 'red']
    
    plt.figure(figsize=(8, 6))
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Maternal Risk Stratification')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved ROC Curve -> {save_path}")

def generate_report():
    print("📊 Loading Data & Model...")
    prep = DataPreprocessor()
    X_fused, y_fused = prep.fuse_datasets()
    X_clin, X_ctg, X_act, X_img = X_fused
    y_risk, y_weight = y_fused
    
    model = build_multimodal_model(
        (X_clin.shape[1],), 
        (X_ctg.shape[1], X_ctg.shape[2]), 
        (X_act.shape[1], X_act.shape[2]), 
        (128, 128, 1)
    )
    
    print(f"   Loading weights from: {model_path}")
    model.load_weights(model_path)
    
    print("   ... Running Predictions on Full Dataset")
    preds = model.predict([X_clin, X_ctg, X_act, X_img], verbose=0)
    
    pred_risk_probs = preds[0]
    pred_risk_classes = np.argmax(pred_risk_probs, axis=1)
    pred_weight = preds[1].flatten()
    
    plot_confusion_matrix(y_risk, pred_risk_classes)
    plot_multiclass_roc(y_risk, pred_risk_probs)
    plot_weight_scatter(y_weight, pred_weight)
    
    print("\n✅ All graphs generated in the 'output' folder!")

if __name__ == "__main__":
    try:
        generate_report()
    except Exception as e:
        print(f"\n❌ Error: {e}")