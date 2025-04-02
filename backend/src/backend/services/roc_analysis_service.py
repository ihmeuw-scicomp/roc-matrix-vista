import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from sqlalchemy.orm import Session
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

from backend.models.roc_data import ROCAnalysis, ConfusionMatrix
from backend.db import get_db
from backend.services.metrics_utils import compute_confusion_matrix, compute_roc_curve, compute_auc, calculate_roc_auc

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
                lambda row: 1 - row[conf_col] if row[label_col].lower() in ['no', 'exclude', 'negative'] else row[conf_col],
                axis=1
            )
    
    return result_df

def add_prediction(df, threshold=0.5):
    """
    Add binary prediction columns based on adjusted confidence and threshold.
    """
    result_df = df.copy()
    
    # Find all adjusted confidence columns
    adjusted_cols = [col for col in result_df.columns if col.endswith('_confidence_adjusted')]
    
    for adj_col in adjusted_cols:
        # Create prediction column
        prefix = adj_col.replace('_confidence_adjusted', '')
        pred_col = prefix + '_prediction'
        
        # Apply threshold
        result_df[pred_col] = result_df[adj_col].apply(lambda x: 1 if x >= threshold else 0)
    
    return result_df

def create_roc_analysis(
    name: str,
    description: str,
    true_labels: List,
    predicted_probs: List,
    default_threshold: float = 0.5,
    unlabeled_predictions: List = None,
    id: Optional[int] = None,
    db: Session = None
) -> ROCAnalysis:
    """
    Create a new ROC analysis or update an existing one.
    
    Args:
        name: Name of the analysis
        description: Description of the analysis
        true_labels: List of true binary labels
        predicted_probs: List of predicted probabilities
        default_threshold: Default threshold for classification
        unlabeled_predictions: List of predictions for unlabeled data (optional)
        id: ID of existing analysis to update (optional)
        db: Database session (optional)
    
    Returns:
        Created or updated ROCAnalysis
    """
    # Convert inputs to numpy arrays for calculation
    y_true = np.array(true_labels)
    y_score = np.array(predicted_probs)
    
    # Calculate ROC curve and AUC
    roc_points, auc_score = calculate_roc_auc(y_true, y_score)
    
    # Initialize the ROCAnalysis object
    roc_analysis = ROCAnalysis(
        id=id,
        name=name,
        description=description,
        true_labels=true_labels,
        predicted_probs=predicted_probs,
        unlabeled_predictions=unlabeled_predictions,
        default_threshold=default_threshold,
        roc_curve_data=roc_points,
        auc_score=auc_score
    )
    
    # Compute confusion matrix for default threshold
    cm_data = compute_confusion_matrix(y_true, y_score, default_threshold)
    
    # Add confusion matrix
    confusion_matrix_obj = ConfusionMatrix(
        roc_analysis_id=id,  # This will be updated after commit if id is None
        threshold=default_threshold,
        **cm_data
    )
    
    # Add to database if session provided
    if db is not None:
        if id is not None:
            # Check if analysis exists
            existing = db.query(ROCAnalysis).filter(ROCAnalysis.id == id).first()
            if existing:
                # Update existing
                for key, value in vars(roc_analysis).items():
                    if key != "_sa_instance_state":
                        setattr(existing, key, value)
                roc_analysis = existing
            else:
                # Create new with specified id
                db.add(roc_analysis)
        else:
            # Create new with auto-generated id
            db.add(roc_analysis)
        
        db.commit()
        db.refresh(roc_analysis)
        
        # Now that we have the analysis id, update and save the confusion matrix
        confusion_matrix_obj.roc_analysis_id = roc_analysis.id
        db.add(confusion_matrix_obj)
        db.commit()
    
    return roc_analysis