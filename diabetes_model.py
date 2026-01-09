# diabetes_model_complete.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                           roc_curve, auc, precision_recall_curve, average_precision_score)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("DIABETES MODEL TRAINING WITH VISUALIZATIONS AND mAP")
print("="*60)

# Load or create sample data
try:
    data = pd.read_csv("diabetes.csv")
    print("✅ Loaded diabetes.csv")
except:
    print("⚠️ Creating sample data...")
    np.random.seed(42)
    n_samples = 768
    data = pd.DataFrame({
        'Pregnancies': np.random.randint(0, 17, n_samples),
        'Glucose': np.random.randint(0, 199, n_samples),
        'BloodPressure': np.random.randint(0, 122, n_samples),
        'SkinThickness': np.random.randint(0, 99, n_samples),
        'Insulin': np.random.randint(0, 846, n_samples),
        'BMI': np.random.uniform(0, 67.1, n_samples),
        'DiabetesPedigreeFunction': np.random.uniform(0.08, 2.42, n_samples),
        'Age': np.random.randint(21, 81, n_samples),
        'Outcome': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
    })

print(f"📊 Dataset shape: {data.shape}")
print(f"📈 Class distribution:\n{data['Outcome'].value_counts()}")

# Prepare data
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Data split complete:")
print(f"   Training: {X_train.shape}")
print(f"   Testing:  {X_test.shape}")

# Train model
print("\n🔧 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    min_samples_split=5,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability for class 1 (Diabetes)

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# ===========================================
# mAP (MEAN AVERAGE PRECISION) CALCULATION
# ===========================================
print("\n🎯 Calculating mAP (Mean Average Precision)...")

def calculate_map_scores(y_true, y_pred_proba, thresholds=np.arange(0.1, 1.0, 0.1)):
    """
    Calculate mAP scores for binary classification
    
    Args:
        y_true: True labels (0 or 1)
        y_pred_proba: Predicted probabilities for class 1
        thresholds: List of thresholds to compute AP at
    
    Returns:
        dict: Dictionary containing mAP and per-threshold metrics
    """
    map_metrics = {}
    
    # 1. Calculate Average Precision (AP) at different thresholds
    precisions_at_recall = {}
    
    for threshold in thresholds:
        # Convert probabilities to binary predictions at this threshold
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        
        # Calculate precision and recall
        TP = np.sum((y_pred_threshold == 1) & (y_true == 1))
        FP = np.sum((y_pred_threshold == 1) & (y_true == 0))
        FN = np.sum((y_pred_threshold == 0) & (y_true == 1))
        
        precision = TP / (TP + FP + 1e-10)  # Avoid division by zero
        recall = TP / (TP + FN + 1e-10)
        
        if recall in precisions_at_recall:
            precisions_at_recall[recall] = max(precision, precisions_at_recall[recall])
        else:
            precisions_at_recall[recall] = precision
    
    # 2. Calculate mAP using 11-point interpolation (PASCAL VOC style)
    recall_levels = np.linspace(0, 1, 11)
    interpolated_precisions = []
    
    for recall_level in recall_levels:
        # Find all precisions at recall >= current level
        valid_precisions = []
        for rec, prec in precisions_at_recall.items():
            if rec >= recall_level:
                valid_precisions.append(prec)
        
        if valid_precisions:
            interpolated_precisions.append(max(valid_precisions))
        else:
            interpolated_precisions.append(0)
    
    # 3. Calculate mAP as mean of interpolated precisions
    mAP = np.mean(interpolated_precisions)
    
    # 4. Store metrics
    map_metrics['mAP'] = mAP
    map_metrics['interpolated_precisions'] = interpolated_precisions
    map_metrics['recall_levels'] = recall_levels
    map_metrics['precisions_at_recall'] = precisions_at_recall
    
    return map_metrics

# Calculate mAP scores
map_results = calculate_map_scores(y_test.values, y_pred_proba)

print(f"   mAP Score: {map_results['mAP']:.4f}")
print(f"   mAP Interpretation:")
print(f"     • mAP > 0.9: Excellent")
print(f"     • mAP > 0.7: Good")
print(f"     • mAP > 0.5: Fair")
print(f"     • mAP < 0.5: Needs Improvement")

# ===========================================
# 1. CONFUSION MATRIX
# ===========================================
print("\n📊 1. Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Diabetes Prediction - Confusion Matrix', fontsize=16, pad=20)
plt.ylabel('Actual Diagnosis', fontsize=12)
plt.xlabel('Predicted Diagnosis', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   💾 Saved: confusion_matrix.png")

# ===========================================
# 2. ROC CURVE (Receiver Operating Characteristic)
# ===========================================
print("\n📈 2. Generating ROC Curve...")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Diabetes Prediction', fontsize=16, pad=20)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# Add optimal threshold point (Youden's J statistic)
youden_j = tpr - fpr
optimal_idx = np.argmax(youden_j)
optimal_threshold = thresholds[optimal_idx]
plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
         label=f'Optimal Threshold: {optimal_threshold:.3f}')

plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("   💾 Saved: roc_curve.png")

# ===========================================
# 3. PRECISION-RECALL CURVE WITH mAP
# ===========================================
print("\n📊 3. Generating Precision-Recall Curve with mAP...")
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
average_precision = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(12, 9))
# Main PR curve
plt.plot(recall, precision, color='darkgreen', lw=3, 
         label=f'PR Curve (AP = {average_precision:.3f})', alpha=0.8)

# Plot mAP points (11-point interpolation)
recall_levels = map_results['recall_levels']
interpolated_precisions = map_results['interpolated_precisions']

plt.scatter(recall_levels, interpolated_precisions, color='red', s=100, 
           zorder=5, label=f'mAP = {map_results["mAP"]:.3f}', marker='X')

# Draw lines for mAP calculation
for i, (rec, prec) in enumerate(zip(recall_levels, interpolated_precisions)):
    plt.plot([rec, rec], [0, prec], 'r--', alpha=0.3, lw=1)
    if i < len(recall_levels) - 1:
        plt.plot([rec, recall_levels[i+1]], [prec, prec], 'r--', alpha=0.3, lw=1)

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve with mAP - Diabetes Prediction', fontsize=16, pad=20)
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend(loc="lower left", fontsize=11)
plt.grid(True, alpha=0.3)

# Add no-skill line
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='gray', 
         label=f'No Skill (AP = {no_skill:.3f})', lw=2)

plt.tight_layout()
plt.savefig('precision_recall_map_curve.png', dpi=300, bbox_inches='tight')
print("   💾 Saved: precision_recall_map_curve.png")

# ===========================================
# 4. mAP DETAILED VISUALIZATION
# ===========================================
print("\n📊 4. Generating mAP Detailed Visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('mAP (Mean Average Precision) Analysis - Diabetes Prediction', 
             fontsize=18, y=1.02)

# Subplot 1: mAP Interpolation Points
ax1 = axes[0, 0]
ax1.step(recall_levels, interpolated_precisions, where='post', 
         color='darkred', lw=2, label='11-point Interpolation')
ax1.scatter(recall_levels, interpolated_precisions, color='red', s=80, zorder=5)
ax1.fill_between(recall_levels, 0, interpolated_precisions, alpha=0.2, color='red')
ax1.set_xlabel('Recall', fontsize=11)
ax1.set_ylabel('Precision', fontsize=11)
ax1.set_title(f'mAP Calculation (mAP = {map_results["mAP"]:.3f})', fontsize=13)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Subplot 2: Precision-Recall at Different Thresholds
ax2 = axes[0, 1]
thresholds_to_plot = np.arange(0.1, 1.0, 0.1)
colors = plt.cm.viridis(np.linspace(0, 1, len(thresholds_to_plot)))

for i, threshold in enumerate(thresholds_to_plot):
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    cm_thresh = confusion_matrix(y_test, y_pred_thresh)
    TN, FP, FN, TP = cm_thresh.ravel()
    prec = TP / (TP + FP + 1e-10)
    rec = TP / (TP + FN + 1e-10)
    ax2.scatter(rec, prec, color=colors[i], s=100, label=f'Thresh={threshold:.1f}')

ax2.set_xlabel('Recall', fontsize=11)
ax2.set_ylabel('Precision', fontsize=11)
ax2.set_title('Precision-Recall at Different Thresholds', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Subplot 3: Threshold vs mAP Components
ax3 = axes[1, 0]
thresholds = np.linspace(0.1, 0.9, 9)
map_components = []

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    cm_thresh = confusion_matrix(y_test, y_pred_thresh)
    TN, FP, FN, TP = cm_thresh.ravel()
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    map_components.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

df_components = pd.DataFrame(map_components)
ax3.plot(df_components['threshold'], df_components['precision'], 'b-', label='Precision', lw=2)
ax3.plot(df_components['threshold'], df_components['recall'], 'g-', label='Recall', lw=2)
ax3.plot(df_components['threshold'], df_components['f1'], 'r-', label='F1-Score', lw=2)
ax3.set_xlabel('Threshold', fontsize=11)
ax3.set_ylabel('Score', fontsize=11)
ax3.set_title('Metrics vs Threshold', fontsize=13)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: mAP Comparison Visualization
ax4 = axes[1, 1]
categories = ['Random\nClassifier', 'Your Model', 'Perfect\nClassifier']
map_values = [0.5, map_results['mAP'], 1.0]
colors_map = ['lightgray', 'skyblue', 'lightgreen']

bars = ax4.bar(categories, map_values, color=colors_map, edgecolor='black', linewidth=2)
ax4.set_ylabel('mAP Score', fontsize=11)
ax4.set_title('mAP Comparison', fontsize=13)
ax4.set_ylim([0, 1.1])

# Add value labels on bars
for bar, val in zip(bars, map_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

# Add performance level
performance_level = ""
if map_results['mAP'] >= 0.9:
    performance_level = "Excellent"
elif map_results['mAP'] >= 0.7:
    performance_level = "Good"
elif map_results['mAP'] >= 0.5:
    performance_level = "Fair"
else:
    performance_level = "Needs Improvement"
    
ax4.text(1, map_results['mAP'] + 0.1, f'Performance: {performance_level}', 
         ha='center', fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
         facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('map_detailed_analysis.png', dpi=300, bbox_inches='tight')
print("   💾 Saved: map_detailed_analysis.png")

# ===========================================
# 5. LEARNING CURVE
# ===========================================
print("\n📚 5. Generating Learning Curve...")
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 8))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training Examples", fontsize=12)
plt.ylabel("Accuracy Score", fontsize=12)
plt.title("Learning Curve - Diabetes Prediction Model", fontsize=16, pad=20)
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
print("   💾 Saved: learning_curve.png")

# ===========================================
# 6. FEATURE IMPORTANCE
# ===========================================
print("\n🎯 6. Generating Feature Importance...")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Feature Importance - Diabetes Prediction', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("   💾 Saved: feature_importance.png")

# ===========================================
# 7. ALL METRICS SUMMARY (INCLUDING mAP)
# ===========================================
print("\n" + "="*60)
print("SUMMARY OF ALL METRICS (INCLUDING mAP)")
print("="*60)

# Calculate all metrics
TN, FP, FN, TP = cm.ravel()

print(f"\n📊 Confusion Matrix:")
print(f"   True Negatives (TN):  {TN:4d}")
print(f"   False Positives (FP): {FP:4d}")
print(f"   False Negatives (FN): {FN:4d}")
print(f"   True Positives (TP):  {TP:4d}")

print(f"\n📈 Performance Metrics:")
print(f"   Accuracy:           {(TP+TN)/(TP+TN+FP+FN):.4f}")
print(f"   Precision:          {TP/(TP+FP):.4f}")
print(f"   Recall (Sensitivity): {TP/(TP+FN):.4f}")
print(f"   F1-Score:           {2*TP/(2*TP+FP+FN):.4f}")
print(f"   Specificity:        {TN/(TN+FP):.4f}")
print(f"   ROC AUC:            {roc_auc:.4f}")
print(f"   Average Precision:  {average_precision:.4f}")
print(f"   mAP (11-point):     {map_results['mAP']:.4f}")

print(f"\n🎯 mAP Interpretation:")
if map_results['mAP'] >= 0.9:
    print(f"   ✅ Excellent performance - Model is highly reliable")
elif map_results['mAP'] >= 0.7:
    print(f"   👍 Good performance - Model performs well")
elif map_results['mAP'] >= 0.5:
    print(f"   ⚠️  Fair performance - Model needs some improvement")
else:
    print(f"   ❌ Poor performance - Model needs significant improvement")

print(f"\n🎯 Top 3 Important Features:")
for i, row in feature_importance.tail(3).iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

print(f"\n📊 mAP Detailed Scores (11-point interpolation):")
for i, (rec, prec) in enumerate(zip(recall_levels, interpolated_precisions)):
    print(f"   Recall={rec:.1f}: Precision={prec:.4f}")

print(f"\n💾 Model saved as: diabetes_model.pkl")
joblib.dump(model, "diabetes_model.pkl")

print(f"\n📁 Generated Visualizations:")
print(f"   1. confusion_matrix.png")
print(f"   2. roc_curve.png")
print(f"   3. precision_recall_map_curve.png")
print(f"   4. map_detailed_analysis.png")
print(f"   5. learning_curve.png")
print(f"   6. feature_importance.png")

print(f"\n✅ All visualizations and mAP analysis generated successfully!")
print("="*60)

# Show all plots
plt.show()