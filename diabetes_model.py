import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("diabetes.csv")

print("✅ Loading dataset...")
print(f"   Shape: {data.shape}")
print(f"   Columns: {data.columns.tolist()}")

# Prepare features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✅ Data split:")
print(f"   Training set: {X_train.shape}")
print(f"   Test set: {X_test.shape}")

# Train Random Forest model
print("\n✅ Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# Generate confusion matrix
print("\n✅ Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix Values:\n{cm}")

# Print classification report
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

# ===========================================
# IMPORTANT: CREATE CONFUSION MATRIX IMAGE
# ===========================================
print("\n📊 Creating confusion matrix image...")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, 
            annot=True, 
            fmt='d',           # 'd' for integers
            cmap='Blues',
            cbar=True,
            square=True,
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])

plt.title('Diabetes Prediction Confusion Matrix\nRandom Forest Classifier', fontsize=16, pad=20)
plt.ylabel('Actual Diagnosis', fontsize=12)
plt.xlabel('Predicted Diagnosis', fontsize=12)
plt.xticks(fontsize=11, rotation=0)
plt.yticks(fontsize=11, rotation=0)

# Add accuracy text
plt.text(0.5, -0.15, 
         f'Accuracy: {accuracy:.2%} | Test Samples: {len(y_test)}',
         ha='center', va='center',
         transform=plt.gca().transAxes,
         fontsize=11,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))

plt.tight_layout()

# Save the image
plt.savefig('confusion_matrix.png', 
            dpi=300, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')

plt.savefig('confusion_matrix_highres.png', 
            dpi=600, 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')

print("💾 Confusion matrix saved as:")
print("   - confusion_matrix.png (300 DPI)")
print("   - confusion_matrix_highres.png (600 DPI)")

# Also save as PDF for documentation
plt.savefig('confusion_matrix.pdf', 
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none')
print("   - confusion_matrix.pdf")

# Show the plot
plt.show()

# Save model
joblib.dump(model, "diabetes_model.pkl")
print("\n💾 Model saved as diabetes_model.pkl")

# Print detailed metrics
TN, FP, FN, TP = cm.ravel()

print("\n📈 Detailed Metrics:")
print(f"True Negatives (TN): {TN} - Correctly predicted no diabetes")
print(f"False Positives (FP): {FP} - Incorrectly predicted diabetes")
print(f"False Negatives (FN): {FN} - Missed diabetes cases")
print(f"True Positives (TP): {TP} - Correctly predicted diabetes")
print(f"\nPrecision (Diabetes): {TP/(TP+FP):.3f}")
print(f"Recall/Sensitivity: {TP/(TP+FN):.3f}")
print(f"Specificity: {TN/(TN+FP):.3f}")
print(f"F1-Score: {2*TP/(2*TP+FP+FN):.3f}")