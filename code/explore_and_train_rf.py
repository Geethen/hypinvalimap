import geopandas as gpd
import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    cohen_kappa_score, 
    confusion_matrix, 
    f1_score
)

# ==============================================================================
# CONFIGURATION & ENVIRONMENT SETUP
# ==============================================================================

# Define path mapping for different environments
PATHS = {
    "cluster": {
        "train": r"/home/geethen/invasives/hypinvalimap/data/2023_wessels_extracted.geojson",
        "eval": r"/home/geethen/invasives/hypinvalimap/data/2023_wessels_HSI_evalpts.geojson"
    },
    "local": {
        "train": r"c:\Users\coach\myfiles\postdoc\Invasives\code\hypinvalimap\data\2023_wessels_extracted.geojson",
        "eval": r"c:\Users\coach\myfiles\postdoc\Invasives\code\hypinvalimap\data\2023_wessels_HSI_evalpts.geojson"
    }
}

def detect_environment():
    """Auto-detects environment (Cluster vs Local) based on path existence."""
    print("\n[STEP 0] Detecting Environment...")
    if os.path.exists(PATHS["cluster"]["train"]):
        print(">>> Environment Detected: CLUSTER")
        return "cluster"
    else:
        print(">>> Environment Detected: LOCAL (Fallback)")
        return "local"

# ==============================================================================
# PREPROCESSING UTILITIES
# ==============================================================================

def remove_problem_bands(gdf):
    """
    Remove spectral bands corresponding to atmospheric absorption/problem regions.
    Ref: HS_experiments_12082025.ipynb
    """
    # Separate wavelength columns (numeric strings) from metadata
    wavelength_cols = [col for col in gdf.columns if col.replace('.', '', 1).isdigit()]
    
    if not wavelength_cols:
        print("    [!] No numeric wavelength columns found. Skipping band removal.")
        return gdf

    wavelengths = np.array(wavelength_cols, dtype=float)

    # Atmospheric mask: < 450nm, 1340-1480nm, 1800-1980nm, > 2000nm
    mask = (
        (wavelengths < 450) |
        ((wavelengths >= 1340) & (wavelengths <= 1480)) |
        ((wavelengths >= 1800) & (wavelengths <= 1980)) |
        (wavelengths > 2000)
    )

    cols_to_drop = np.array(wavelength_cols)[mask]
    gdf_filtered = gdf.drop(columns=cols_to_drop)
    
    print(f"    [Preprocessing] Dropped {len(cols_to_drop)} problem bands.")
    print(f"    [Preprocessing] Bands preserved: {len(wavelength_cols) - len(cols_to_drop)}")
    return gdf_filtered

def preprocess_dataset(gdf, name="Dataset"):
    """
    General data cleaning and target column preparation.
    """
    print(f"\n[STEP] Preprocessing {name}...")
    
    # 1. Update 2023 class based on change detection logic
    if 'change' in gdf.columns and 'class' in gdf.columns:
        if '2023_class' not in gdf.columns:
            gdf['2023_class'] = np.nan
        
        # for all points that did not change, set the 2023_class to the 2018 class
        unchanged_mask = gdf['change'] == 0
        update_count = unchanged_mask.sum()
        gdf.loc[unchanged_mask, '2023_class'] = gdf['class']
        print(f"    [Preprocessing] Inherited 2018 classes for {update_count} unchanged points.")

    # 2. Determine Target Column
    target_col = '2023_class' if '2023_class' in gdf.columns else 'class'
    print(f"    [Preprocessing] Using '{target_col}' as target response.")

    # 3. Filter Class IDs (Ref: IRMAD_experiments requires ID < 12)
    if gdf[target_col].dtype in [np.float64, np.int64]:
        initial_len = len(gdf)
        gdf = gdf[gdf[target_col] < 12]
        print(f"    [Preprocessing] Filtered classes < 12: {initial_len} -> {len(gdf)} rows.")

    # 4. Filter Spectral Bands
    gdf = remove_problem_bands(gdf)

    return gdf, target_col

# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def run_experiment():
    # 1. Environment and File Verification
    env = detect_environment()
    train_path = PATHS[env]["train"]
    eval_path = PATHS[env]["eval"]

    if not os.path.exists(train_path):
        print(f"CRITICAL ERROR: Training file not found at {train_path}")
        sys.exit(1)

    # 2. Loading and Preprocessing Training Data
    print(f"\n[STEP 1] Loading Training Data from: {train_path}")
    train_gdf = gpd.read_file(train_path)
    train_gdf, target_col = preprocess_dataset(train_gdf, name="TRAIN")

    # 3. Feature Identification
    # Assume any column that is a numeric wavelength is a feature
    feature_cols = [col for col in train_gdf.columns if col.replace('.', '', 1).isdigit()]
    if not feature_cols:
        print("    [!] No wavelength bands detected. Falling back to numeric columns.")
        exclude = ['geometry', 'class', '2023_class', 'change', 'index_right', 'dist', 'id', 'POINT_X', 'POINT_Y']
        feature_cols = [c for c in train_gdf.columns if c not in exclude and train_gdf[c].dtype in [np.float64, np.int64]]
    
    print(f"    [Features] Total features identified: {len(feature_cols)}")

    # 4. Final Training Set Preparation
    train_data = train_gdf[feature_cols + [target_col]].dropna()
    print(f"    [Stats] Final training samples (no NaNs): {len(train_data)}")
    print(f"    [Stats] Class Distribution:\n{train_data[target_col].value_counts().sort_index()}")

    X_train = train_data[feature_cols]
    y_train = train_data[target_col].astype(int)

    # 5. Model Training
    print(f"\n[STEP 2] Training Random Forest Classifier...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(">>> Model Training Complete.")

    # 6. Loading and Preprocessing Evaluation Data
    if os.path.exists(eval_path):
        print(f"\n[STEP 3] Loading Evaluation Data from: {eval_path}")
        eval_gdf = gpd.read_file(eval_path)
        eval_gdf, _ = preprocess_dataset(eval_gdf, name="EVAL")

        # Use same features as selected in training
        eval_data = eval_gdf[feature_cols + [target_col]].dropna()
        
        if len(eval_data) == 0:
            print("    [!] No evaluation data remaining after cleaning. Check if bands/classes match.")
        else:
            print(f"    [Stats] Final evaluation samples: {len(eval_data)}")
            print(f"    [Stats] Eval Class Distribution:\n{eval_data[target_col].value_counts().sort_index()}")

            X_eval = eval_data[feature_cols]
            y_eval = eval_data[target_col].astype(int)

            # 7. Model Evaluation
            print(f"\n[STEP 4] Executing External Performance Evaluation...")
            y_pred = rf.predict(X_eval)

            # Metrics Calculation
            acc = accuracy_score(y_eval, y_pred)
            f1 = f1_score(y_eval, y_pred, average='weighted')
            kappa = cohen_kappa_score(y_eval, y_pred)

            print("-" * 40)
            print(f"OVERALL PERFORMANCE METRICS")
            print("-" * 40)
            print(f"Accuracy: {acc:.4f}")
            print(f"F1 Score (Weighted): {f1:.4f}")
            print("-" * 40)

            print("\nConfusion Matrix:")
            print(confusion_matrix(y_eval, y_pred))

            print("\nDetailed Classification Report:")
            print(classification_report(y_eval, y_pred))
    else:
        print(f"\n[STEP 3] SKIPPED: Evaluation file not found at {eval_path}")

if __name__ == "__main__":
    run_experiment()
