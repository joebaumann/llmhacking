import pdb
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import ast
from data.data_utils import map_dataset_name_to_class


output_fp = ['results_FINAL']


def load_llm_annotated_data(task_name, output_filepath=None):
    """Load all CSV files for a given task from multiple paths"""
    if output_filepath is None:
        output_filepath = output_fp  # Default to list with single path
    if isinstance(output_filepath, str):
        output_filepath = [output_filepath]  # Convert single string to list
    
    all_dfs = []
    
    # Process each path in the list
    for filepath in output_filepath:
        if not Path(filepath).exists():
            print(f"Warning: Directory does not exist: {filepath}")
            pdb.set_trace()

        task_results_path = Path(filepath, task_name)
        
        if not task_results_path.exists():
            print(f"Warning: Path does not exist: {task_results_path}")
            # pdb.set_trace()
            continue
        
        all_csv_results_files = [f for f in list(task_results_path.glob('*.csv')) if 'batch_metadata' not in str(f)]
        
        # Find all CSV files in the task directory (not in subdirectories)
        for csv_file in tqdm(all_csv_results_files, desc=f'Processing csv files from {filepath}', leave=True, total=len(all_csv_results_files)):
            # Read CSV, auto-detect headers
            try:
                df = pd.read_csv(csv_file, header='infer')
            except pd.errors.EmptyDataError:
                # Handle empty file case
                print(f"Warning: skipping empty csv file: {csv_file}")
                continue
            except Exception as e:
                print(f"Could not load csv file: {e}")
                df = pd.DataFrame()
                continue
            
            # Check if first row might be header
            expected_cols = ['id', 'ground_truth', 'response_mapped', 'response', 
                            'output_mapping_description', 'model', 'params', 'prompt_description']
            if not all(col in df.columns for col in expected_cols[:3]):
                # Set header as used during experiments
                if (len(df.columns) - len(expected_cols)) == 2:
                    cols = expected_cols + ['seed', 'temperature']
                elif (len(df.columns) - len(expected_cols)) == 1:
                    cols = expected_cols + ['temperature']
                elif len(df.columns) == len(expected_cols):
                    cols = expected_cols
                else:
                    print('Need to check df columns')
                    pdb.set_trace()
                df = pd.read_csv(csv_file, header=None, names=cols)
            
            # Add timestamp from filename
            timestamp = csv_file.stem
            df['timestamp'] = timestamp
            
            # Add source path to track where data came from
            df['source_path'] = str(filepath)
            
            # Extract temperature and seed from params if not in columns
            try:
                if 'temperature' not in df.columns:
                    df['temperature'] = df['params'].apply(
                        lambda x: ast.literal_eval(x).get('temperature', 0.0) if pd.notna(x) else 0.0
                    )
            except Exception as e:
                print(e)
                pdb.set_trace()
                df['temperature'] = 0.0
            
            try:
                # Extract seed from params (always do this as seed is not a column)
                df['seed'] = df['params'].apply(
                    lambda x: ast.literal_eval(x).get('seed', 42) if pd.notna(x) else 42
                )
            except Exception as e:
                print(e)
                pdb.set_trace()
                df['seed'] = 42
            
            # Convert data types
            if 'ground_truth' in df.columns:
                df['ground_truth'] = df['ground_truth'].astype(str)
            if 'response_mapped' in df.columns:
                df['response_mapped'] = df['response_mapped'].astype(str)

            all_dfs.append(df)

    # Concatenate all dataframes
    if all_dfs:
        all_dfs = pd.concat(all_dfs, ignore_index=True)
    else:
        print(f'No data found for task: {task_name}')
        pdb.set_trace()
        all_dfs = pd.DataFrame()

    # Get all ids from the dataset
    data_loader = map_dataset_name_to_class(task_name)
    dataset = data_loader.load_dataset()
    all_task_ids = list(dataset['id'].unique())
    
    # Store original count before filtering
    original_count = len(all_dfs)
    
    # Filter to only include IDs that exist in the dataset
    all_dfs = all_dfs[all_dfs['id'].isin(all_task_ids)]
    
    # Calculate and report removed count
    removed_count = original_count - len(all_dfs)
    if removed_count > 0:
        print(f"Removed {removed_count} rows with IDs that are not in the dataset's IDs.")
    
    return all_dfs


def binarize_labels(data, class_name, cols_to_binarize=None):
    if cols_to_binarize is None:
        cols_to_binarize = ['ground_truth', 'ground_truth_known', 'response_mapped']
    
    # Create a copy to avoid modifying the original data
    data_copy = data.copy()
    
    for col in cols_to_binarize:
        if col in data_copy.columns:
            if set(data_copy[col].unique()) == {0,1}:
                # column already binarized
                continue
            # Convert to string, strip whitespace, and convert to lowercase for comparison
            normalized_col = data_copy[col].astype(str).str.strip().str.lower()
            normalized_class = str(class_name).lower()
            
            # Overwrite existing column with binary values (1 for positive class, 0 for negative)
            data_copy[col] = (normalized_col == normalized_class).astype(int)
    
    return data_copy

def get_class_imbalance(dataset, col_name='temp_class_column'):
    all_classes = dataset[col_name].value_counts()

    # check if binary labels
    if len(all_classes) == 2:
        # Get counts for majority and minority classes
        majority_count = all_classes.max()
        minority_count = all_classes.min()
        imbalance_ratio = (majority_count - minority_count) / (len(dataset))
    else:
        # throw not implemented error
        raise NotImplementedError("Function only supports binary classification problems")
    
    return imbalance_ratio



def load_prepared_prompts(prompts_base_path, task_name, model):
    """Load prepared confidence prompts for a specific task and model."""
    model_name = model.replace('/', '--')
    prompts_file = Path(prompts_base_path, task_name, model_name, 'all_prompts_for_confidence_elicitation.csv')
    
    if not prompts_file.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(prompts_file)
        return df
    except Exception as e:
        print(f"Error loading prompts file {prompts_file}: {e}")
        pdb.set_trace()
        return pd.DataFrame()

def load_confidence_csv_file(csv_file):
    df = pd.read_csv(csv_file)
    
    # Check if we have the header mismatch issue
    if len(df.columns) == 6 and 'response_mapped' not in df.columns:
        # This is the "broken" format - headers are missing 'response_mapped'
        # The actual data has 7 columns but header only has 6
        
        # Re-read with correct column names
        correct_columns = [
            'id',
            'response_mapped', 
            'confidence',
            'model',
            'params',
            'seed',
            'temperature'
        ]
        
        df = pd.read_csv(csv_file, names=correct_columns, skiprows=1)
        # print("Loaded CSV with corrected headers (added missing 'response_mapped' column)")
        
    elif 'response_mapped' in df.columns:
        # This is the correct format
        # print("Loaded CSV with correct headers")
        pass
        
    else:
        # Unexpected format
        print(f"Warning: Unexpected CSV format with columns: {list(df.columns)}")
        pdb.set_trace()
    
    return df
            



def load_confidence_results(results_base_path, task_name, model):
    """Load confidence elicitation results for a specific task and model."""
    model_name = model.replace('/', '--')
    
    # Try to find results files (CSV files with timestamps)
    task_model_path = Path(results_base_path, task_name, model_name)
    
    if not task_model_path.exists():
        return pd.DataFrame()
    
    # Look for CSV files (excluding runtimes and prompts)
    csv_files = [f for f in task_model_path.glob('*.csv') 
                 if 'runtime' not in f.name.lower() 
                 and 'prompt' not in f.name.lower()]
    
    if not csv_files:
        return pd.DataFrame()
    
    # Load all CSV files and concatenate
    all_dfs = []
    for csv_file in csv_files:
        try:
            df = load_confidence_csv_file(csv_file)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            pdb.set_trace()
            continue
    
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()
