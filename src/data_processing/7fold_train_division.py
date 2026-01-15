import pandas as pd
import os
from pathlib import Path


def divide_train_into_7folds():
    """
    Divides the training data from train.csv into 7 folds of approximately equal size
    and saves them in the 7folds directory.
    """
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    train_csv_path = project_root / "src" / "dataset" / "splits" / "train.csv"
    output_dir = project_root / "src" / "dataset" / "7folds"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the training data
    print(f"Reading training data from {train_csv_path}...")
    df = pd.read_csv(train_csv_path)
    
    # Get the number of rows
    num_rows = len(df)
    num_folds = 7
    
    # Calculate fold sizes
    rows_per_fold = num_rows // num_folds
    remainder = num_rows % num_folds
    
    print(f"Total rows: {num_rows}")
    print(f"Dividing into {num_folds} folds...")
    
    # Split the data into folds
    start_idx = 0
    for fold_num in range(1, num_folds + 1):
        # Add one extra row to the first 'remainder' folds to distribute remainder evenly
        fold_size = rows_per_fold + (1 if fold_num <= remainder else 0)
        end_idx = start_idx + fold_size
        
        # Extract fold data
        fold_df = df.iloc[start_idx:end_idx].copy()
        
        # Save fold to CSV
        output_path = output_dir / f"fold_{fold_num}.csv"
        fold_df.to_csv(output_path, index=False)
        
        print(f"  Fold {fold_num}: {len(fold_df)} rows -> {output_path}")
        
        # Update start index for next fold
        start_idx = end_idx
    
    print(f"\nSuccessfully divided training data into {num_folds} folds!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    divide_train_into_7folds()

