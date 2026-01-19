# Import necessary libraries
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import fbeta_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.pipeline import Pipeline
import seaborn as sns

from ..utils.data_loader import load_split, prepare_features_target
from ..models.manage_models import save_model

# Set random state for reproducibility
RANDOM_STATE = 42

# Load data
print("Loading data...")
train_df = load_split('train', data_dir='../dataset/splits')
test_df = load_split('test', data_dir='../dataset/splits')

# Prepare features and target
X_train, y_train = prepare_features_target(train_df, target_col='Fault')
X_test, y_test = prepare_features_target(test_df, target_col='Fault')

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Class distribution in training set: {np.bincount(y_train)}")
print(f"Class distribution in test set: {np.bincount(y_test)}")

# Define SVM parameters (from Optuna optimization)
svm_params = {
    'kernel': 'rbf',
    'C': 69.84841896499474,
    'gamma': 'auto',
    'probability': True,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',
    'cache_size': 2000,
    'tol': 1e-3
}

# Create pipeline with StandardScaler and SVM
print("\nCreating and training final SVM model...")
final_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC(**svm_params))
])

# Train the model on all training data
final_pipeline.fit(X_train, y_train)
print("Model training completed!")

paths = save_model(
    model=final_pipeline, 
    model_name='svm_final',
    save_path='models/',
    metadata={
        'augmented': False,
        'undersampled': False,
        'scaler_applied': True,
        'hyperparameters': final_pipeline.get_params()
    }
)
print(f"Modelo guardado en: {paths['model_path']}")

# Make predictions on test set
print("Evaluating model on test set...")
y_pred = final_pipeline.predict(X_test)
y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
f2_score = fbeta_score(y_test, y_pred, beta=2)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Generate classification report
class_report = classification_report(y_test, y_pred, output_dict=True)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate Precision-Recall curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)

# Calculate feature statistics for visualization
feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'Feature_{i}' for i in range(X_train.shape[1])]

# Calculate means by class (for feature analysis visualization)
if len(X_train[y_train == 0]) > 0:
    means_class_0 = X_train[y_train == 0].mean().values.tolist()
else:
    means_class_0 = [0] * len(feature_names)

if len(X_train[y_train == 1]) > 0:
    means_class_1 = X_train[y_train == 1].mean().values.tolist()
else:
    means_class_1 = [0] * len(feature_names)

# Calculate class distributions
train_class_dist = np.bincount(y_train).tolist()
test_class_dist = np.bincount(y_test).tolist()

# Calculate additional statistics
error_rate = 1 - accuracy
specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0

# Prediction distribution
pred_counts = pd.Series(y_pred).value_counts().sort_index()
true_counts = pd.Series(y_test).value_counts().sort_index()

# Print results
print("\n" + "="*50)
print("FINAL MODEL PERFORMANCE ON TEST SET")
print("="*50)
print(f"Accuracy: {accuracy:.4f}")
print(f"F2-Score: {f2_score:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Error Rate: {error_rate:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Store results in a dictionary
results = {
    'model_name': 'SVM',
    'parameters': svm_params,
    'test_metrics': {
        'accuracy': float(accuracy),
        'f2_score': float(f2_score),
        'precision': float(precision),
        'recall': float(recall),
        'roc_auc': float(roc_auc),
        'specificity': float(specificity),
        'error_rate': float(error_rate)
    },
    'cross_validation': {
        'best_score': 0.8788507336559883,  # From Optuna optimization
        'folds': 10,
        'metric': 'f2_score'
    },
    'confusion_matrix': cm.tolist(),
    'classification_report': class_report,
    'dataset_info': {
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'train_class_distribution': train_class_dist,
        'test_class_distribution': test_class_dist,
        'feature_names': feature_names,
        'feature_statistics': {
            'means_class_0': means_class_0,
            'means_class_1': means_class_1
        }
    },
    'curve_data': {
        'roc_curve': {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist()
        },
        'pr_curve': {
            'precision': precision_vals.tolist(),
            'recall': recall_vals.tolist()
        }
    },
    'prediction_distribution':{
        'pred_counts': pred_counts,
        'true_counts': true_counts
    },
    'paths': {
        'train_data': '../dataset/splits/train.csv',
        'test_data': '../dataset/splits/test.csv',
        'model_file': str(paths.get('model_path', ''))
    }
}

# Save results to JSON file
results_dir = 'results'
import os
os.makedirs(results_dir, exist_ok=True)

results_file = os.path.join(results_dir, 'svm_real_results.json')
with open(results_file, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to: {results_file}")

# Also save as a more readable format
readable_results = f"""
SVM MODEL FINAL RESULTS
{'='*50}

MODEL PARAMETERS:
{json.dumps(svm_params, indent=2)}

DATASET INFORMATION:
- Training samples: {len(X_train)}
- Test samples: {len(X_test)}
- Training class distribution: {np.bincount(y_train).tolist()}
- Test class distribution: {np.bincount(y_test).tolist()}

PERFORMANCE METRICS ON TEST SET:
- Accuracy: {accuracy:.4f}
- F2-Score: {f2_score:.4f}
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- ROC-AUC: {roc_auc:.4f}

CONFUSION MATRIX:
{cm}

CLASSIFICATION REPORT:
{classification_report(y_test, y_pred)}
"""

# Save readable version
readable_file = os.path.join(results_dir, 'svm_real_results_readable.txt')
with open(readable_file, 'w') as f:
    f.write(readable_results)

print(f"Readable results saved to: {readable_file}")