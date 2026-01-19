# Import necessary libraries
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import fbeta_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.pipeline import Pipeline
import os
from typing import Tuple, Optional, Dict, Any
import joblib
from datetime import datetime


def load_split(
    split_name: str,
    data_dir: str = "src/dataset/splits"
) -> pd.DataFrame:
    """
    Load a dataset split (train, validation, or test).
    
    Parameters:
    -----------
    split_name : str
        Name of the split to load ('train', 'validation', or 'test')
    data_dir : str
        Directory containing the split files
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataset split
        
    Raises:
    -------
    FileNotFoundError
        If the split file doesn't exist
    ValueError
        If split_name is not one of the valid options
    """
    valid_splits = ['train', 'validation', 'test']
    if split_name not in valid_splits:
        raise ValueError(f"split_name must be one of {valid_splits}, got '{split_name}'")
    
    file_path = os.path.join(data_dir, f"{split_name}.csv")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Split file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    return df


def prepare_features_target(
    df: pd.DataFrame,
    target_col: str = "Fault"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target from a dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing features and target
    target_col : str
        Name of the target column
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        Tuple of (features, target)
        
    Raises:
    -------
    KeyError
        If target_col is not found in the dataframe
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return X, y

def save_model(
    model: Any,
    model_name: str,
    save_path: str = 'src/models/',
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Save a trained model and its metadata.
    
    Parameters:
    -----------
    model : Any
        Trained model to save
    model_name : str
        Name identifier for the model
    save_path : str, default='src/models/'
        Directory to save the model
    metadata : dict, optional
        Additional metadata to save (hyperparameters, metrics, etc.)
        
    Returns:
    --------
    dict
        Dictionary with paths to saved model and metadata files
        
    Examples:
    ---------
    >>> paths = save_model(model, 'logistic_regression', metadata={'accuracy': 0.95})
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_filename = f"{model_name}_{timestamp}.joblib"
    model_path = os.path.join(save_path, model_filename)
    joblib.dump(model, model_path)
    
    # Prepare metadata
    model_metadata = {
        'model_name': model_name,
        'model_path': model_path,
        'timestamp': timestamp,
        'training_date': datetime.now().isoformat(),
    }
    
    # Add model-specific attributes if available
    if hasattr(model, 'get_params'):
        model_metadata['hyperparameters'] = model.get_params()
    
    # Add custom metadata
    if metadata:
        model_metadata.update(metadata)
    
    # Save metadata
    metadata_filename = f"{model_name}_{timestamp}_metadata.json"
    metadata_path = os.path.join(save_path, metadata_filename)
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2, default=str)
    
    return {
        'model_path': model_path,
        'metadata_path': metadata_path
    }

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
print(f"Class distribution in training set: {y_train.value_counts().sort_index().tolist()}")
print(f"Class distribution in test set: {y_test.value_counts().sort_index().tolist()}")

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
train_class_dist = y_train.value_counts().sort_index().tolist()
test_class_dist = y_test.value_counts().sort_index().tolist()

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
        'pred_counts': pred_counts.tolist(),
        'true_counts': true_counts.tolist()
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
- Training class distribution: {y_train.value_counts().sort_index().tolist()}
- Test class distribution: {y_test.value_counts().sort_index().tolist()}

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
