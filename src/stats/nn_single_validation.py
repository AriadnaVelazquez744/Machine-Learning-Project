import pandas as pd
import numpy as np
import os
import pickle
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import json
import time

# Machine Learning Imports
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # Special pipeline for sampling

# Statistical Test Imports
from scipy import stats

# Visualization (optional)
import matplotlib.pyplot as plt
import seaborn as sns

# Configure display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.set_printoptions(suppress=True)

# Global Constants
N_FOLDS = 10
RANDOM_STATE = 42
BETA = 2  # For F2-score

# Define paths
FOLDS_DIR = '../dataset/folds/'
STATS_DIR = 'results'
MODELS_DIR = 'models/'

# Create directories if they don't exist
for directory in [STATS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Define model names
MODEL_NAMES = ['SVM', 'XGBoost', 'RandomForest', 'NeuralNetwork']

# Data structures to store results
@dataclass
class FoldResults:
    """Store results from a single fold validation"""
    fold_num: int
    f2_score: float
    f1_score: float  # Added for statistical comparison
    accuracy: float  # Added for statistical comparison
    precision: float
    recall: float
    conf_matrix: List[List[int]]  # [[TN, FP], [FN, TP]]
    train_time: float
    pred_time: float
    train_samples: int
    test_samples: int
    y_true: List[float] = None  # Optional: true labels for McNemar test
    y_pred: List[float] = None  # Optional: predictions for McNemar test

@dataclass
class ModelValidationResults:
    """Store all validation results for a single model"""
    model_name: str
    hyperparameters: Dict[str, Any]
    fold_results: List[FoldResults]
    
    # Calculated metrics
    f2_scores: List[float]
    precision_scores: List[float]
    recall_scores: List[float]
    f1_scores: List[float] = None  # Added for statistical comparison
    accuracy_scores: List[float] = None  # Added for statistical comparison
    
    def __post_init__(self):
        """Initialize lists if not provided"""
        if self.f1_scores is None:
            self.f1_scores = [fr.f1_score for fr in self.fold_results]
        if self.accuracy_scores is None:
            self.accuracy_scores = [fr.accuracy for fr in self.fold_results]
    
    @property
    def mean_f2(self) -> float:
        return np.mean(self.f2_scores) if self.f2_scores else 0.0
    
    @property
    def std_f2(self) -> float:
        return np.std(self.f2_scores) if self.f2_scores else 0.0
    
    @property
    def mean_f1(self) -> float:
        return np.mean(self.f1_scores) if self.f1_scores else 0.0
    
    @property
    def std_f1(self) -> float:
        return np.std(self.f1_scores) if self.f1_scores else 0.0
    
    @property
    def mean_accuracy(self) -> float:
        return np.mean(self.accuracy_scores) if self.accuracy_scores else 0.0
    
    @property
    def std_accuracy(self) -> float:
        return np.std(self.accuracy_scores) if self.accuracy_scores else 0.0
    
    @property
    def mean_precision(self) -> float:
        return np.mean(self.precision_scores) if self.precision_scores else 0.0
    
    @property
    def mean_recall(self) -> float:
        return np.mean(self.recall_scores) if self.recall_scores else 0.0
    
    def calculate_confusion_matrices_sum(self) -> Tuple[int, int, int, int]:
        """Sum confusion matrices from all folds"""
        total_tn = total_fp = total_fn = total_tp = 0
        for fold_result in self.fold_results:
            tn, fp, fn, tp = fold_result.conf_matrix[0][0], fold_result.conf_matrix[0][1], \
                            fold_result.conf_matrix[1][0], fold_result.conf_matrix[1][1]
            total_tn += tn
            total_fp += fp
            total_fn += fn
            total_tp += tp
        return total_tn, total_fp, total_fn, total_tp
    
    def save_to_file(self, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            filename = f"{self.model_name}_validation_results.json"
        
        filepath = os.path.join(STATS_DIR, filename)
        
        # Convert to dictionary
        results_dict = {
            'model_name': self.model_name,
            'hyperparameters': self.hyperparameters,
            'metrics_summary': {
                'mean_f2': float(self.mean_f2),
                'std_f2': float(self.std_f2),
                'mean_f1': float(self.mean_f1),
                'std_f1': float(self.std_f1),
                'mean_accuracy': float(self.mean_accuracy),
                'std_accuracy': float(self.std_accuracy),
                'mean_precision': float(self.mean_precision),
                'mean_recall': float(self.mean_recall),
                'f2_scores': [float(score) for score in self.f2_scores],
                'f1_scores': [float(score) for score in self.f1_scores],
                'accuracy_scores': [float(score) for score in self.accuracy_scores],
                'precision_scores': [float(score) for score in self.precision_scores],
                'recall_scores': [float(score) for score in self.recall_scores],
            },
            'fold_details': [
                {
                    'fold_num': fr.fold_num,
                    'f2_score': float(fr.f2_score),
                    'f1_score': float(fr.f1_score),
                    'accuracy': float(fr.accuracy),
                    'precision': float(fr.precision),
                    'recall': float(fr.recall),
                    'conf_matrix': fr.conf_matrix,
                    'train_time': float(fr.train_time),
                    'pred_time': float(fr.pred_time),
                    'train_samples': fr.train_samples,
                    'test_samples': fr.test_samples,
                    'y_true': (fr.y_true.tolist() if hasattr(fr.y_true, 'tolist') else list(fr.y_true)) if fr.y_true is not None else None,
                    'y_pred': (fr.y_pred.tolist() if hasattr(fr.y_pred, 'tolist') else list(fr.y_pred)) if fr.y_pred is not None else None
                }
                for fr in self.fold_results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {filepath}")

# Define hyperparameters for each model (from your report)
MODEL_HYPERPARAMETERS = {
    'SVM': {
        'kernel': 'rbf',
        'C': 69.84841896499474,
        'gamma': 'auto',
        'probability': True,
        'random_state': RANDOM_STATE,
        'class_weight': 'balanced'  # From final optimized SVM
    },
    
    'XGBoost': {
        'objective': 'binary:logistic',
        'max_depth': 10,
        'learning_rate': 0.0273,
        'min_child_weight': 17,
        'gamma': 3.459e-7,           # Lagrangian multiplier (min_split_loss)
        'reg_lambda': 7.739,         # L2 regularization (λ)
        'reg_alpha': 0.0067,         # L1 regularization (α)
        'subsample': 0.997,
        'colsample_bytree': 0.986,
        'scale_pos_weight': 2.438,
        'tree_method': 'hist',       # As used in the tuned XGBoost notebook
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss'
    },
    
    'RandomForest': {
        'n_estimators': 179,
        'max_depth': 19,
        'min_samples_leaf': 3,
        'random_state': RANDOM_STATE
        # Sampling handled externally via RandomUnderSampler
    },
    
    'NeuralNetwork': {
        'hidden_layer_sizes': (32, 128),  # (32, 128) as in the best Optuna trial
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 1.249e-5,               # Regularization
        'learning_rate_init': 0.00851,
        'max_iter': 148,
        'random_state': RANDOM_STATE,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 10,
        'batch_size': 'auto'
    }
}

# Define sampling strategies for each model
MODEL_SAMPLING_STRATEGIES = {
    'SVM': None,           # Best SVM uses no external sampling
    'XGBoost': None,       # XGBoost uses scale_pos_weight for class imbalance
    'RandomForest': 'undersample',  # Random Under Sampling
    'NeuralNetwork': 'smote'        # SMOTE oversampling
}

# Function to load folds
def load_folds(folds_dir: str = FOLDS_DIR) -> List[pd.DataFrame]:
    """Load all fold CSVs into a list"""
    folds = []
    for i in range(1, N_FOLDS + 1):
        fold_path = os.path.join(folds_dir, f'fold_{i}.csv')
        if os.path.exists(fold_path):
            fold_df = pd.read_csv(fold_path)
            folds.append(fold_df)
            print(f"Loaded fold {i}: {fold_df.shape}")
        else:
            print(f"Warning: Fold {i} not found at {fold_path}")
            folds.append(pd.DataFrame())
    return folds

# Function to get train/test indices for a given test fold
def get_fold_split(folds: List[pd.DataFrame], test_fold_idx: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get training and testing data for a given fold index"""
    test_data = folds[test_fold_idx]
    
    # Combine all other folds for training
    train_folds = [folds[i] for i in range(len(folds)) if i != test_fold_idx]
    train_data = pd.concat(train_folds, ignore_index=True)
    
    return train_data, test_data

# Function to split features and target
def split_features_target(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split data into features (X) and target (y)"""
    # Assuming target column is 'Fault' based on your report
    target_col = 'Fault'
    
    if target_col not in data.columns:
        # Try to find it case-insensitively
        possible_cols = [col for col in data.columns if col.lower() == 'fault']
        if possible_cols:
            target_col = possible_cols[0]
        else:
            raise ValueError(f"Target column 'Fault' not found. Available columns: {data.columns.tolist()}")
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    return X, y

# Function to calculate F2-score
def calculate_f2_score(y_true, y_pred, beta: int = BETA) -> float:
    """Calculate F-beta score with given beta"""
    return fbeta_score(y_true, y_pred, beta=beta)

# Load all folds once (will be reused for all models)
print("Loading folds...")
ALL_FOLDS = load_folds()

# Verify folds were loaded correctly
if len(ALL_FOLDS) == 0:
    print("ERROR: No folds loaded. Check the folds directory.")
elif len(ALL_FOLDS) != N_FOLDS:
    print(f"WARNING: Expected {N_FOLDS} folds, but loaded {len(ALL_FOLDS)}")
    
print(f"\nGlobal variables initialized for {N_FOLDS}-fold cross-validation")
print(f"Models to validate: {MODEL_NAMES}")
print(f"Results will be saved to: {STATS_DIR}")

'''
Function to make one cycle of the validation of the Neural Networks model depending on the fold number pass as param
'''
def validate_nn_fold(fold_idx: int):
    """
    Validate Neural Networks model for a single fold.
    
    Args:
        fold_idx: Index of the fold to validate (0-based, 0-9 for 10 folds)
    
    This function:
    - Validates the specified fold
    - Loads existing results from file if available
    - Updates or adds the fold result
    - Saves all results back to file
    """

    if fold_idx < 0 or fold_idx >= N_FOLDS:
        raise ValueError(f"fold_idx must be between 0 and {N_FOLDS-1}, got {fold_idx}")
    
    print("="*70)
    print(f"Neural Networks MODEL - FOLD {fold_idx + 1} VALIDATION")
    print("="*70)
    
    # Get Neural Network hyperparameters and sampling strategy
    nn_params = MODEL_HYPERPARAMETERS['NeuralNetwork'].copy()
    nn_sampling = MODEL_SAMPLING_STRATEGIES['NeuralNetwork']

    print(f"\nHyperparameters: {nn_params}")
    print(f"Sampling strategy: {nn_sampling}")
    
    # Get train/test split for this fold
    train_data, test_data = get_fold_split(ALL_FOLDS, fold_idx)
    
    # Split features and target
    X_train, y_train = split_features_target(train_data)
    X_test, y_test = split_features_target(test_data)
    
    print(f"\nFold {fold_idx + 1}/{N_FOLDS}")
    print("-"*70)
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    # Create Neural Network model with hyperparameters
    nn_model = MLPClassifier(**nn_params)
    
    # Create pipeline: StandardScaler + Neural Network (SMOTE sampling applied externally)
    nn_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Escalado obligatorio para redes neuronales
        ('clf', nn_model)
    ])

    # Train the model
    train_start = time.time()
    nn_pipeline.fit(X_train, y_train)
    train_time = time.time() - train_start
    
    # Make predictions
    pred_start = time.time()
    y_pred = nn_pipeline.predict(X_test)
    pred_time = time.time() - pred_start
    
    # Calculate metrics
    f2 = calculate_f2_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Convert confusion matrix to list format [[TN, FP], [FN, TP]]
    cm_list = [[int(cm[0, 0]), int(cm[0, 1])], [int(cm[1, 0]), int(cm[1, 1])]]
    
    # Store fold results (including predictions for McNemar test)
    fold_result = FoldResults(
        fold_num=fold_idx + 1,
        f2_score=f2,
        f1_score=f1,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        conf_matrix=cm_list,
        train_time=train_time,
        pred_time=pred_time,
        train_samples=len(X_train),
        test_samples=len(X_test),
        y_true=y_test.values if hasattr(y_test, 'values') else np.array(y_test),
        y_pred=y_pred
    )
    
    print(f"\nResults for Fold {fold_idx + 1}:")
    print(f"F2-Score: {f2:.4f}, F1-Score: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Train time: {train_time:.2f}s, Prediction time: {pred_time:.4f}s")
    print(f"Confusion Matrix:\n{cm}")

    # Load existing results or create new
    filename = "Neural_Networks_validation_results.json"
    filepath = os.path.join(STATS_DIR, filename)
    
    if os.path.exists(filepath):
        # Load existing results
        with open(filepath, 'r') as f:
            existing_data = json.load(f)
        
        # Extract existing fold results
        existing_fold_details = existing_data.get('fold_details', [])
        
        # Remove the fold if it already exists (to update it)
        existing_fold_details = [fd for fd in existing_fold_details if fd['fold_num'] != fold_idx + 1]
        
        # Add the new fold result
        new_fold_detail = {
            'fold_num': fold_result.fold_num,
            'f2_score': float(fold_result.f2_score),
            'f1_score': float(fold_result.f1_score),
            'accuracy': float(fold_result.accuracy),
            'precision': float(fold_result.precision),
            'recall': float(fold_result.recall),
            'conf_matrix': fold_result.conf_matrix,
            'train_time': float(fold_result.train_time),
            'pred_time': float(fold_result.pred_time),
            'train_samples': fold_result.train_samples,
            'test_samples': fold_result.test_samples,
            'y_true': (fold_result.y_true.tolist() if hasattr(fold_result.y_true, 'tolist') else list(fold_result.y_true)) if fold_result.y_true is not None else None,
            'y_pred': (fold_result.y_pred.tolist() if hasattr(fold_result.y_pred, 'tolist') else list(fold_result.y_pred)) if fold_result.y_pred is not None else None
        }
        existing_fold_details.append(new_fold_detail)
        
        # Reconstruct FoldResults objects from all folds
        all_fold_results = []
        for fd in existing_fold_details:
            all_fold_results.append(FoldResults(
                fold_num=fd['fold_num'],
                f2_score=fd['f2_score'],
                f1_score=fd['f1_score'],
                accuracy=fd['accuracy'],
                precision=fd['precision'],
                recall=fd['recall'],
                conf_matrix=fd['conf_matrix'],
                train_time=fd['train_time'],
                pred_time=fd['pred_time'],
                train_samples=fd['train_samples'],
                test_samples=fd['test_samples'],
                y_true=np.array(fd['y_true']) if fd['y_true'] is not None else None,
                y_pred=np.array(fd['y_pred']) if fd['y_pred'] is not None else None
            ))
        
        # Extract all scores
        all_f2_scores = [fr.f2_score for fr in all_fold_results]
        all_f1_scores = [fr.f1_score for fr in all_fold_results]
        all_accuracy_scores = [fr.accuracy for fr in all_fold_results]
        all_precision_scores = [fr.precision for fr in all_fold_results]
        all_recall_scores = [fr.recall for fr in all_fold_results]
        
        # Create ModelValidationResults with all folds
        nn_validation_results = ModelValidationResults(
            model_name='NeuralNetwork',
            hyperparameters=nn_params,
            fold_results=all_fold_results,
            f2_scores=all_f2_scores,
            precision_scores=all_precision_scores,
            recall_scores=all_recall_scores,
            f1_scores=all_f1_scores,
            accuracy_scores=all_accuracy_scores
        )
        
        print(f"\nUpdated existing results. Total folds validated: {len(all_fold_results)}/{N_FOLDS}")
    else:
        # Create new results with just this fold
        nn_validation_results = ModelValidationResults(
            model_name='NeuralNetwork',
            hyperparameters=nn_params,
            fold_results=[fold_result],
            f2_scores=[f2],
            precision_scores=[precision],
            recall_scores=[recall],
            f1_scores=[f1],
            accuracy_scores=[accuracy]
        )
        print(f"\nCreated new results file. Folds validated: 1/{N_FOLDS}")
    
    # Save results to file (using fixed filename so all folds save to same file)
    nn_validation_results.save_to_file(filename)
    
    # Print summary if we have multiple folds
    if len(nn_validation_results.fold_results) > 1:
        print("\n" + "="*70)
        print("CURRENT Neural Networks CROSS-VALIDATION SUMMARY")
        print("="*70)
        print(f"Folds completed: {len(nn_validation_results.fold_results)}/{N_FOLDS}")
        print(f"Mean F2-Score: {nn_validation_results.mean_f2:.4f} ± {nn_validation_results.std_f2:.4f}")
        print(f"Mean F1-Score: {nn_validation_results.mean_f1:.4f} ± {nn_validation_results.std_f1:.4f}")
        print(f"Mean Accuracy: {nn_validation_results.mean_accuracy:.4f} ± {nn_validation_results.std_accuracy:.4f}")
        print(f"Mean Precision: {nn_validation_results.mean_precision:.4f}")
        print(f"Mean Recall: {nn_validation_results.mean_recall:.4f}")
        print(f"\nF2-Scores per fold: {[f'{s:.4f}' for s in nn_validation_results.f2_scores]}")
    
    print(f"\nFold {fold_idx + 1} validation completed. Results saved to {filepath}")
    return nn_validation_results

if __name__ == "__main__":
    # Manually determine the fold to validate
    # Change this variable to the fold index you want to validate (0-9 for 10 folds)
    # fold_idx is 0-based: 0 = fold 1, 1 = fold 2, ..., 9 = fold 10
    # Already processed folds: 1, 2, 3, 4, 5, 6, 
    for i in [6, 7, 8, 9]:
        # FOLD_TO_VALIDATE = i  # Change this value to validate different folds
        print(f"Validating fold call {i}")
        # Validate the specified fold
        # nn_validation_results = validate_nn_fold(FOLD_TO_VALIDATE)
        nn_validation_results = validate_nn_fold(i)
