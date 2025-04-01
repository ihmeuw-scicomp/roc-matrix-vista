import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from sqlalchemy.orm import Session
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

from backend.models.roc_data import ROCAnalysis, ConfusionMatrix
from backend.db import get_db

def num_iter(col_lst):
    """
    Determine the number of iterations from column names by finding the maximum
    numeric suffix in the column names.
    
    Args:
        col_lst: List of column names
        
    Returns:
        int: Maximum iteration number
    """
    num_lst = []
    for name in col_lst:
        suffix = name.split('_')[-1]
        if suffix.isdigit() and len(suffix) < 2:
            num_lst.append(suffix)
    
    return int(max(num_lst)) if num_lst else 0

def get_column(data_frame):
    """
    Returns the prefixes of columns that have 'confidence' in their name.
    
    Parameters:
    data_frame (pd.DataFrame): Input DataFrame
    
    Returns:
    list: List of unique prefixes from column names containing 'confidence'
    """
    # Find all columns with 'confidence' in their name
    confidence_cols = [col for col in data_frame.columns if 'confidence' in col.lower()]
    
    # Handle case where no confidence columns exist
    if not confidence_cols:
        return []
    
    # Extract prefixes from all confidence columns
    prefixes = []
    for col in confidence_cols:
        parts = col.split('_confidence_')
        if len(parts) > 0:
            prefixes.append(parts[0])
    
    # Return unique prefixes
    return list(set(prefixes))

def avg_conf_correction(majority_label, conf_df, name_df):
    """
    Calculate adjusted average confidence based on majority label.
    
    Args:
        majority_label: The majority label in the data
        conf_df: List of confidence scores
        name_df: List of labels
        
    Returns:
        float: Adjusted average confidence score
    """
    numerator = []
    for i in range(len(conf_df)):
        if majority_label != name_df[i]: 
            conf_num = 1 - conf_df[i]
            numerator.append(conf_num)
        else:
            conf_num = conf_df[i]
            numerator.append(conf_num)
    return sum(numerator) / len(conf_df) if len(conf_df) > 0 else 0

def avg_con(name, num, data_frame, config=None):
    """
    Compute average confidence and majority label for a set of columns.
    
    Args:
        name: Base name of the columns
        num: Number of iterations
        data_frame: Pandas DataFrame with a single row
        config: Configuration dictionary with optional settings
        
    Returns:
        tuple: (average_confidence, majority_label)
    """
    # Initialize config with defaults if not provided
    if config is None:
        config = {
            'default_confidence': 0.5,
            'default_label': 'unknown'
        }
    
    name_lst = [name + '_' + str(x+1) for x in range(num)]
    conf_lst = [name + '_confidence_' + str(x+1) for x in range(num)]
    
    # Check if all required columns exist
    missing_cols = [col for col in name_lst + conf_lst if col not in data_frame.columns]
    if missing_cols:
        # Return default values if columns are missing
        return (config['default_confidence'], config['default_label'])
    
    # Get the majority label
    all_labels = [data_frame[col].iloc[0].strip() for col in name_lst]
    majority_label = Counter(all_labels).most_common(1)[0][0]
    
    # Handle missing confidence values
    if data_frame.loc[:, conf_lst].isnull().any(axis=1).values[0]:
        mean_conf = data_frame.loc[:, conf_lst].mean(axis=1, skipna=True).values[0]
        new_frame = data_frame.loc[:, conf_lst].fillna(mean_conf)
        conf_df = new_frame.loc[:, conf_lst].values[0]
        name_df = data_frame.loc[:, name_lst].values[0]
        name_df = [i.strip() for i in name_df]
    else:
        conf_df = data_frame.loc[:, conf_lst].values[0]
        name_df = data_frame.loc[:, name_lst].values[0]
        name_df = [i.strip() for i in name_df]

    avg_confidence = avg_conf_correction(majority_label, conf_df, name_df)
    return (avg_confidence, majority_label)

def find_column_by_suffix(df, suffix):
    """
    Find a column in a DataFrame that ends with a specific suffix.
    
    Args:
        df: DataFrame to search in
        suffix: Suffix string to match
        
    Returns:
        str: First column name that matches the suffix, or None if not found
    """
    matching_columns = [col for col in df.columns if col.endswith(suffix)]
    return matching_columns[0] if matching_columns else None

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process the input dataframe to standardize column names and values"""
    # Copy the dataframe to avoid modifying the original
    processed_df = df.copy()
    
    # Handle common operations like standardizing column names
    processed_df.columns = [col.strip() for col in processed_df.columns]
    
    return processed_df

def process_dataframe(df, method='average', config=None):
    """
    Process a DataFrame with dynamic column detection and configurable settings.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    method (str): Processing method ('average', 'majority', etc.)
    config (dict): Configuration dictionary with optional settings
    
    Returns:
    pd.DataFrame: Processed DataFrame
    """
    # Initialize config with defaults if not provided
    if config is None:
        config = {
            'confidence_pattern': 'confidence',
            'label_pattern': '_label',
            'positive_labels': ['yes', 'Include', 'positive'],
            'threshold': 0.5,
            'default_confidence': 0.5,
            'default_label': 'unknown',
            'confidence_suffix': '_confidence',
            'label_suffix': '_label',
            'adjusted_suffix': '_confidence_adjusted',
            'prediction_suffix': '_confidence_prediction'
        }
    
    result_df = df.copy()
    
    # Dynamically detect column groups
    column_groups = detect_column_groups(result_df, config['confidence_pattern'])
    
    # Process each column group based on method
    for prefix, iter_count in column_groups.items():
        # Apply the specified method
        if method.lower() == 'average':
            # Add confidence column
            result_df[prefix + config['confidence_suffix']] = result_df.apply(
                lambda row: avg_con(prefix, iter_count, pd.DataFrame([row]), config)[0], 
                axis=1
            )
            # Add label column
            result_df[prefix + config['label_suffix']] = result_df.apply(
                lambda row: avg_con(prefix, iter_count, pd.DataFrame([row]), config)[1], 
                axis=1
            )
        elif method.lower() == 'majority':
            # Implement majority method if needed
            pass
        # Add other methods as needed
    
    # Adjust probabilities based on labels
    result_df = adjust_probabilities(result_df, config)
    
    # Add binary prediction column
    result_df = add_prediction(result_df, config['threshold'])
    
    return result_df

def detect_column_groups(df, confidence_pattern):
    """
    Dynamically detect column groups and their iteration counts.
    
    Returns:
    dict: {prefix: iteration_count} for each group
    """
    groups = {}
    confidence_cols = [col for col in df.columns if confidence_pattern in col.lower()]
    
    for col in confidence_cols:
        parts = col.split('_' + confidence_pattern + '_')
        if len(parts) == 2 and parts[1].isdigit():
            prefix = parts[0]
            num = int(parts[1])
            groups[prefix] = max(groups.get(prefix, 0), num)
    
    return groups

def adjust_probabilities(df, config):
    """
    Adjust probabilities based on labels to ensure they represent 
    the probability of the positive class.
    """
    result_df = df.copy()
    
    # Find all label columns
    label_cols = [col for col in result_df.columns if col.endswith(config['label_suffix'])]
    
    for label_col in label_cols:
        # Find matching confidence column - extract prefix before the label_suffix
        prefix = label_col[:-len(config['label_suffix'])]
        conf_col = prefix + config['confidence_suffix']
        
        if conf_col in result_df.columns:
            # Ensure confidence values are numeric
            result_df[conf_col] = pd.to_numeric(result_df[conf_col], errors='coerce').fillna(config['default_confidence'])
            
            # Create adjusted confidence column
            adj_col = prefix + config['adjusted_suffix']
            result_df[adj_col] = result_df.apply(
                lambda row: row[conf_col] 
                if row[label_col].strip().lower() in [l.lower() for l in config['positive_labels']]
                else 1 - row[conf_col], 
                axis=1
            )
    
    return result_df

def add_prediction(df, threshold=0.5):
    """
    Add binary prediction columns based on adjusted confidence
    """
    result_df = df.copy()
    
    # Find all columns ending with '_confidence_adjusted'
    adjusted_cols = [col for col in result_df.columns if '_confidence_adjusted' in col]
    
    for col in adjusted_cols:
        # Replace the suffix while keeping the prefix intact
        prefix = col.replace('_confidence_adjusted', '')
        pred_col = prefix + '_confidence_prediction'
        result_df[pred_col] = result_df[col].apply(lambda x: 1 if x >= threshold else 0)
    
    return result_df

def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> List[Dict]:
    """
    Compute ROC curve points using scikit-learn, formatting the result 
    as a list of dictionaries suitable for JSON serialization.
    
    This is primarily a wrapper around sklearn.metrics.roc_curve that formats
    the output for database storage.
    
    Args:
        y_true: Array of true binary labels
        y_score: Array of predicted probabilities
        
    Returns:
        List[Dict]: ROC curve points with threshold, TPR, and FPR values
    """
    # Use sklearn's implementation directly
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # Format the results as a list of dictionaries for database storage, filtering out infinity values
    roc_points = [
        {
            "threshold": float(thresholds[i]),
            "tpr": float(tpr[i]),
            "fpr": float(fpr[i])
        }
        for i in range(len(thresholds)) if not np.isinf(thresholds[i])
    ]
    
    # Add the point (1,1) if it's not already included
    if len(roc_points) > 0 and roc_points[-1]["threshold"] > 0:
        roc_points.append({
            "threshold": 0.0,
            "tpr": 1.0,
            "fpr": 1.0
        })
    
    return roc_points

def compute_auc(roc_points: List[Dict]) -> float:
    """
    Calculate AUC from ROC points using scikit-learn's auc function.
    
    This is primarily a wrapper that extracts FPR and TPR values from
    our dictionary format and passes them to sklearn.metrics.auc.
    
    Args:
        roc_points: List of dictionaries with TPR and FPR values
        
    Returns:
        float: Area Under the ROC Curve
    """
    if not roc_points:
        return 0.0
    
    # Extract FPR and TPR values from our dictionary format
    fpr = [point["fpr"] for point in roc_points]
    tpr = [point["tpr"] for point in roc_points]
    
    # Use scikit-learn's AUC function directly
    return float(auc(fpr, tpr))

# Alternatively, we could create a more direct function that doesn't require the intermediate format
def calculate_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[List[Dict], float]:
    """
    Directly calculate both ROC curve points and AUC score from raw data.
    This eliminates the need for separate compute_roc_curve and compute_auc calls.
    
    Args:
        y_true: Array of true binary labels
        y_score: Array of predicted probabilities
        
    Returns:
        Tuple[List[Dict], float]: (roc_points, auc_score)
    """
    # Get ROC curve data using sklearn
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # Calculate AUC directly
    roc_auc = auc(fpr, tpr)
    
    # Format ROC points for database storage, filtering out infinity values
    roc_points = [
        {
            "threshold": float(thresholds[i]),
            "tpr": float(tpr[i]),
            "fpr": float(fpr[i])
        }
        for i in range(len(thresholds)) if not np.isinf(thresholds[i])
    ]
    
    # Add the point (1,1) if it's not already included
    if len(roc_points) > 0 and roc_points[-1]["threshold"] > 0:
        roc_points.append({
            "threshold": 0.0,
            "tpr": 1.0,
            "fpr": 1.0
        })
    
    return roc_points, float(roc_auc)

# Update create_roc_analysis to use the new function
def create_roc_analysis(
    name: str,
    description: str,
    true_labels: np.ndarray,
    predicted_probs: np.ndarray,
    default_threshold: float = 0.5,
    id: Optional[int] = None,
    db: Session = None
) -> ROCAnalysis:
    """Create and save a new ROC analysis with confusion matrices."""
    if db is None:
        db = next(get_db())
    
    # Use the combined function to get both ROC points and AUC score
    roc_points, auc_score = calculate_roc_auc(true_labels, predicted_probs)
    
    # Create ROC analysis object and store the raw data as lists
    roc_analysis = ROCAnalysis(
        id=id,
        name=name,
        description=description,
        default_threshold=default_threshold,
        roc_curve_data=roc_points,
        auc_score=auc_score,
        true_labels=true_labels.tolist(),  # Store original data for later use
        predicted_probs=predicted_probs.tolist()
    )
    
    # Add to database
    db.add(roc_analysis)
    db.commit()
    db.refresh(roc_analysis)
    
    # Generate confusion matrices for important thresholds
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    # Also include the default threshold if not in list
    if default_threshold not in thresholds:
        thresholds.append(default_threshold)
        thresholds.sort()
    
    for threshold in thresholds:
        cm_data = compute_confusion_matrix(true_labels, predicted_probs, threshold)
        cm = ConfusionMatrix(
            roc_analysis_id=roc_analysis.id,
            threshold=threshold,
            **cm_data
        )
        db.add(cm)
    
    db.commit()
    return roc_analysis

def compute_confusion_matrix(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict:
    """
    Compute confusion matrix and related metrics at a specified threshold.
    
    Args:
        y_true: Array of true binary labels
        y_score: Array of predicted probabilities
        threshold: Threshold to apply for binary classification
        
    Returns:
        Dict: Dictionary containing confusion matrix data and metrics
    """
    # Convert probabilities to binary predictions using threshold
    y_pred = (y_score >= threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate additional metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Handle potential infinity values by replacing them with None or 0
    if np.isinf(precision):
        precision = 0.0
    if np.isinf(recall):
        recall = 0.0
    if np.isinf(f1):
        f1 = 0.0
    if np.isinf(accuracy):
        accuracy = 0.0
    
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }