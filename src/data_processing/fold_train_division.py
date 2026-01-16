import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from collections import Counter

def create_stratified_folds():
    """
    Create stratified 10-fold splits maintaining the class imbalance
    in the fault variable (binary classification).
    """
    # Define paths
    train_path = 'src/dataset/splits/train.csv'
    folds_dir = 'src/dataset/folds/'
    
    # Create folds directory if it doesn't exist
    os.makedirs(folds_dir, exist_ok=True)
    
    # Load training data
    print(f"Loading training data from {train_path}...")
    train_data = pd.read_csv(train_path)
    
    # Check data shape and class distribution
    print(f"Training data shape: {train_data.shape}")
    print(f"Training data columns: {train_data.columns.tolist()}")
    
    # Assuming the target column is named 'Fault' (based on your report)
    target_col = 'Fault'
    
    if target_col not in train_data.columns:
        # Try to find the target column (case-insensitive)
        possible_cols = [col for col in train_data.columns if 'fault' in col.lower()]
        if possible_cols:
            target_col = possible_cols[0]
            print(f"Found target column: {target_col}")
        else:
            print("Available columns:", train_data.columns.tolist())
            target_col = input("Please enter the exact name of the target column: ")
    
    # Check class distribution
    class_distribution = train_data[target_col].value_counts()
    print("\nOriginal class distribution in training set:")
    print(f"Class 0 (Normal): {class_distribution.get(0, 0)} samples ({100*class_distribution.get(0, 0)/len(train_data):.2f}%)")
    print(f"Class 1 (Fault): {class_distribution.get(1, 0)} samples ({100*class_distribution.get(1, 0)/len(train_data):.2f}%)")
    
    # Extract features and target
    X = train_data.drop(columns=[target_col]) if target_col in train_data.columns else train_data
    y = train_data[target_col] if target_col in train_data.columns else None
    
    if y is None:
        raise ValueError(f"Target column '{target_col}' not found in the dataset")
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=43)
    
    # Create folds
    fold_data = []
    fold_info = []
    
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        # Get validation fold data
        fold_df = train_data.iloc[val_idx].copy()
        
        # Calculate class distribution in this fold
        fold_class_dist = Counter(fold_df[target_col])
        fold_info.append({
            'fold': fold_num,
            'samples': len(fold_df),
            'class_0': fold_class_dist.get(0, 0),
            'class_1': fold_class_dist.get(1, 0),
            'pct_class_0': 100 * fold_class_dist.get(0, 0) / len(fold_df),
            'pct_class_1': 100 * fold_class_dist.get(1, 0) / len(fold_df)
        })
        
        # Save fold to CSV
        fold_filename = os.path.join(folds_dir, f'fold_{fold_num}.csv')
        fold_df.to_csv(fold_filename, index=False)
        fold_data.append(fold_df)
        
        print(f"  Fold {fold_num}: {len(fold_df)} samples")
    
    # Test correctness of implementation
    print("\n" + "="*60)
    print("TESTING CORRECTNESS OF IMPLEMENTATION")
    print("="*60)
    
    # Test 1: Check that all folds combined contain all training data
    all_fold_indices = set()
    total_fold_samples = 0
    
    for fold_df in fold_data:
        total_fold_samples += len(fold_df)
    
    print(f"\nTest 1 - Completeness:")
    print(f"  Original training samples: {len(train_data)}")
    print(f"  Total samples in all folds: {total_fold_samples}")
    print(f"  All training data accounted for: {total_fold_samples == len(train_data)}")
    
    # Test 2: Check no overlap between folds
    fold_indices_list = []
    for fold_num, (_, val_idx) in enumerate(skf.split(X, y), 1):
        fold_indices_list.append(set(val_idx))
    
    overlaps_found = False
    for i in range(len(fold_indices_list)):
        for j in range(i+1, len(fold_indices_list)):
            if fold_indices_list[i].intersection(fold_indices_list[j]):
                overlaps_found = True
                print(f"  WARNING: Overlap found between fold {i+1} and fold {j+1}")
    
    if not overlaps_found:
        print(f"  Test 2 - No overlap between folds: PASSED")
    
    # Test 3: Check class distribution in each fold
    print(f"\nTest 3 - Class distribution in each fold:")
    fold_info_df = pd.DataFrame(fold_info)
    print(fold_info_df.to_string(index=False))
    
    # Calculate average class distribution
    avg_class_0 = fold_info_df['pct_class_0'].mean()
    avg_class_1 = fold_info_df['pct_class_1'].mean()
    
    print(f"\nAverage class distribution across all folds:")
    print(f"  Class 0: {avg_class_0:.2f}% (target: {100*class_distribution.get(0, 0)/len(train_data):.2f}%)")
    print(f"  Class 1: {avg_class_1:.2f}% (target: {100*class_distribution.get(1, 0)/len(train_data):.2f}%)")
    
    # Test 4: Check file creation
    print(f"\nTest 4 - File creation:")
    files_created = []
    for fold_num in range(1, 11):
        fold_filename = os.path.join(folds_dir, f'fold_{fold_num}.csv')
        if os.path.exists(fold_filename):
            files_created.append(fold_num)
    
    if len(files_created) == 10:
        print(f"  All 10 fold files created successfully")
    else:
        print(f"  WARNING: Only {len(files_created)} files created")
    
    # Test 5: Verify fold sizes are approximately equal
    print(f"\nTest 5 - Fold size consistency:")
    fold_sizes = [info['samples'] for info in fold_info]
    avg_size = np.mean(fold_sizes)
    size_std = np.std(fold_sizes)
    
    print(f"  Average fold size: {avg_size:.1f} samples")
    print(f"  Standard deviation: {size_std:.1f} samples")
    print(f"  Min fold size: {min(fold_sizes)} samples")
    print(f"  Max fold size: {max(fold_sizes)} samples")
    
    # Calculate expected fold size (should be 10% of data)
    expected_size = len(train_data) / 10
    size_variation = max(abs(size - expected_size) for size in fold_sizes)
    
    if size_variation <= 1:  # Allow for 1 sample variation due to stratification
        print(f"  Fold sizes are approximately equal: PASSED")
    else:
        print(f"  WARNING: Fold sizes vary more than expected")
    
    # Save fold information to a summary file
    summary_path = os.path.join(folds_dir, 'fold_summary.csv')
    fold_info_df.to_csv(summary_path, index=False)
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Training data loaded: {len(train_data)} samples")
    print(f"✓ 10 stratified folds created in: {folds_dir}")
    print(f"✓ Class imbalance preserved (~{100*class_distribution.get(1, 0)/len(train_data):.1f}% fault cases in each fold)")
    print(f"✓ Fold summary saved to: {summary_path}")
    
    return fold_info_df

def load_and_verify_folds():
    """
    Load and verify the created folds.
    Useful for checking the folds in a separate session.
    """
    folds_dir = 'src/dataset/folds/'
    
    if not os.path.exists(folds_dir):
        print(f"Error: Folds directory not found: {folds_dir}")
        return None
    
    fold_files = [f for f in os.listdir(folds_dir) if f.startswith('fold_') and f.endswith('.csv')]
    
    if not fold_files:
        print(f"No fold files found in {folds_dir}")
        return None
    
    print(f"Found {len(fold_files)} fold files")
    
    fold_info = []
    for fold_file in sorted(fold_files):
        fold_path = os.path.join(folds_dir, fold_file)
        fold_df = pd.read_csv(fold_path)
        
        # Extract fold number from filename
        fold_num = int(fold_file.split('_')[1].split('.')[0])
        
        # Assuming target column is 'Fault'
        target_col = 'Fault'
        if target_col not in fold_df.columns:
            # Try to find it
            possible_cols = [col for col in fold_df.columns if 'fault' in col.lower()]
            if possible_cols:
                target_col = possible_cols[0]
        
        if target_col in fold_df.columns:
            class_dist = fold_df[target_col].value_counts()
            fold_info.append({
                'fold': fold_num,
                'samples': len(fold_df),
                'class_0': class_dist.get(0, 0),
                'class_1': class_dist.get(1, 0),
                'pct_class_0': 100 * class_dist.get(0, 0) / len(fold_df),
                'pct_class_1': 100 * class_dist.get(1, 0) / len(fold_df)
            })
        else:
            print(f"Warning: Target column not found in {fold_file}")
            fold_info.append({
                'fold': fold_num,
                'samples': len(fold_df),
                'class_0': 'N/A',
                'class_1': 'N/A',
                'pct_class_0': 'N/A',
                'pct_class_1': 'N/A'
            })
    
    fold_info_df = pd.DataFrame(fold_info)
    print("\nFold Information:")
    print(fold_info_df.to_string(index=False))
    
    return fold_info_df

if __name__ == "__main__":
    # Create the folds
    print("Creating stratified 10-fold splits...")
    print("-" * 60)
    
    try:
        fold_summary = create_stratified_folds()
        
        # Optional: Load and verify folds
        print("\n\nVerifying created folds...")
        print("-" * 60)
        load_and_verify_folds()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the train.csv file exists at 'src/dataset/splits/train.csv'")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check that 'src/dataset/splits/train.csv' exists")
        print("2. Verify the CSV file has the correct format")
        print("3. Check if the target column is named 'Fault' (case-sensitive)")
        print("4. Ensure you have read/write permissions in the directories")