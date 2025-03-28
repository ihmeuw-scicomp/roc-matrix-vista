import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sqlalchemy.orm import Session
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from collections import Counter

from backend.models.roc_data import ROCAnalysis, ConfusionMatrix
from backend.db.session import get_db

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
            'default_label': 'unknown'
        }
    
    result_df = df.copy()
    
    # Dynamically detect column groups
    column_groups = detect_column_groups(result_df, config['confidence_pattern'])
    
    # Process each column group based on method
    for prefix, iter_count in column_groups.items():
        # Apply the specified method
        if method.lower() == 'average':
            # Add confidence column
            result_df[prefix + "_confidence"] = result_df.apply(
                lambda row: avg_con(prefix, iter_count, pd.DataFrame([row]), config)[0], 
                axis=1
            )
            # Add label column
            result_df[prefix + "_label"] = result_df.apply(
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
    label_cols = [col for col in result_df.columns if config['label_pattern'] in col]
    
    for label_col in label_cols:
        # Find matching confidence column
        prefix = label_col.replace(config['label_pattern'], '')
        conf_col = prefix + "_confidence"
        
        if conf_col in result_df.columns:
            # Ensure confidence values are numeric
            result_df[conf_col] = pd.to_numeric(result_df[conf_col], errors='coerce').fillna(config['default_confidence'])
            
            # Create adjusted confidence column
            adj_col = f"{conf_col}_adjusted"
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
    
    for col in result_df.columns:
        if '_adjusted' in col:
            pred_col = col.replace('_adjusted', '_prediction')
            result_df[pred_col] = result_df[col].apply(lambda x: 1 if x >= threshold else 0)
    
    return result_df

def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> List[Dict]:
    """
    Compute ROC curve points using scikit-learn.
    
    Args:
        y_true: Array of true binary labels
        y_score: Array of predicted probabilities
        
    Returns:
        List[Dict]: ROC curve points with threshold, TPR, and FPR values
    """
    # Use sklearn's implementation for efficiency and accuracy
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    # Format the results as a list of dictionaries
    roc_points = []
    for i in range(len(thresholds)):
        roc_points.append({
            "threshold": float(thresholds[i]),
            "tpr": float(tpr[i]),
            "fpr": float(fpr[i])
        })
    
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
    Calculate AUC from ROC points using the trapezoidal rule.
    
    Args:
        roc_points: List of dictionaries with TPR and FPR values
        
    Returns:
        float: Area Under the ROC Curve
    """
    if not roc_points:
        return 0.0
    
    # Extract FPR and TPR values
    fpr = [point["fpr"] for point in roc_points]
    tpr = [point["tpr"] for point in roc_points]
    
    # Use scikit-learn's AUC function for accuracy
    return float(auc(fpr, tpr))

def compute_confusion_matrix(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict:
    """
    Compute confusion matrix metrics for a specific threshold.
    
    Args:
        y_true: Array of true binary labels
        y_score: Array of predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dict: Confusion matrix metrics
    """
    # Convert probabilities to binary predictions using the threshold
    y_pred = (y_score >= threshold).astype(int)
    
    # Calculate confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate derived metrics
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn), 
        "false_negatives": int(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }
def create_roc_analysis(
    name: str,
    description: str,
    true_labels: np.ndarray,
    predicted_probs: np.ndarray,
    default_threshold: float = 0.5,
    db: Session = None
) -> ROCAnalysis:
    """Create and save a new ROC analysis with confusion matrices."""
    if db is None:
        db = next(get_db())
    
    # Calculate ROC curve data
    roc_points = compute_roc_curve(true_labels, predicted_probs)
    auc_score = compute_auc(roc_points)
    
    # Create ROC analysis object
    roc_analysis = ROCAnalysis(
        name=name,
        description=description,
        default_threshold=default_threshold,
        roc_curve_data=roc_points,
        auc_score=auc_score
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
def plot_roc_curve(true_labels: np.ndarray, predicted_probs: np.ndarray) -> plt.Figure:
    """
    Create and return a ROC curve plot.
    
    Args:
        true_labels: Array of true binary labels
        predicted_probs: Array of predicted probabilities
        
    Returns:
        matplotlib.figure.Figure: ROC curve plot
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)
    
    # Find the threshold that maximizes TPR
    max_tpr_index = np.argmax(tpr)
    max_threshold = thresholds[max_tpr_index]
    max_fpr = fpr[max_tpr_index]
    max_tpr = tpr[max_tpr_index]
    
    # Create figure and plot
    fig = plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    
    # Highlight max TPR point
    plt.scatter(max_fpr, max_tpr, color='red', label='Max TPR Threshold', zorder=5)
    plt.annotate(
        f'Threshold: {max_threshold:.2f}\nTPR: {max_tpr:.2f}, FPR: {max_fpr:.2f}',
        xy=(max_fpr, max_tpr),
        xytext=(max_fpr + 0.1, max_tpr - 0.1),
        arrowprops=dict(facecolor='black', arrowstyle='->'),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor='gray', facecolor='white', alpha=0.7)
    )
    
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    return fig

def plot_threshold_curves(true_labels: np.ndarray, predicted_probs: np.ndarray) -> plt.Figure:
    """
    Create and return a plot showing TPR and FPR as functions of threshold.
    
    Args:
        true_labels: Array of true binary labels
        predicted_probs: Array of predicted probabilities
        
    Returns:
        matplotlib.figure.Figure: Threshold curve plot
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probs)
    
    # Create figure and plot
    fig = plt.figure(figsize=(8, 6))
    plt.plot(thresholds, tpr, marker='o', label='TPR')
    plt.plot(thresholds, fpr, marker='o', label='FPR')
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.title("TPR and FPR as a Function of Threshold")
    plt.grid(True)
    plt.legend()
    plt.gca().invert_xaxis()  # Invert x-axis as threshold decreases from left to right
    
    return fig