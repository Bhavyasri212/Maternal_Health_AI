import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from preprocessing import DataPreprocessor
from model import build_multimodal_model
import os

# Create output folder if missing
if not os.path.exists('../output'):
    os.makedirs('../output')

def plot_confusion_matrix(y_true, y_pred_classes):
    """Generates the Confusion Matrix for Risk Classification."""
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'Mid', 'High'], 
                yticklabels=['Low', 'Mid', 'High'])
    plt.xlabel('Predicted Risk')
    plt.ylabel('Actual Risk')
    plt.title('Confusion Matrix: Maternal Risk')
    plt.savefig('../output/confusion_matrix.png')
    print("✅ Saved Confusion Matrix -> output/confusion_matrix.png")

def plot_weight_scatter(y_true, y_pred):
    """Generates a Scatter Plot for Birth Weight (Actual vs Predicted)."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='purple')
    
    # Plot perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    plt.xlabel('Actual Birth Weight (g)')
    plt.ylabel('Predicted Birth Weight (g)')
    plt.title('Regression Analysis: Actual vs Predicted Weight')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('../output/weight_scatter_plot.png')
    print("✅ Saved Weight Scatter Plot -> output/weight_scatter_plot.png")

def plot_multiclass_roc(y_true, y_pred_probs):
    """Generates ROC Curves for Multi-Class Risk."""
    n_classes = 3
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # One-hot encode true labels for ROC calculation
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
    plt.savefig('../output/roc_curve.png')
    print("✅ Saved ROC Curve -> output/roc_curve.png")

def generate_report():
    print("📊 Loading Data & Model...")
    
    # 1. Load Data
    prep = DataPreprocessor()
    X_fused, y_fused = prep.fuse_datasets()
    X_clin, X_ctg, X_act, X_img = X_fused
    y_risk, y_weight = y_fused
    
    # 2. Rebuild Model (Twin-Tower Architecture)
    model = build_multimodal_model(
        (X_clin.shape[1],), 
        (X_ctg.shape[1], X_ctg.shape[2]), 
        (X_act.shape[1], X_act.shape[2]), 
        (128, 128, 1)
    )
    
    # 3. Load Weights
    model.load_weights('../output/best_maternal_model.keras')
    
    # 4. Predict
    print("   ... Running Predictions on Full Dataset")
    preds = model.predict([X_clin, X_ctg, X_act, X_img], verbose=0)
    
    pred_risk_probs = preds[0]          # (N, 3) probabilities
    pred_risk_classes = np.argmax(pred_risk_probs, axis=1) # (N,) class indices
    pred_weight = preds[1].flatten()    # (N,) scalar weights
    
    # 5. Generate Plots
    plot_confusion_matrix(y_risk, pred_risk_classes)
    plot_multiclass_roc(y_risk, pred_risk_probs)
    plot_weight_scatter(y_weight, pred_weight)
    
    print("\n✅ All graphs generated in the 'output' folder!")

if __name__ == "__main__":
    try:
        generate_report()
    except Exception as e:
        print(f"❌ Error: {e}")