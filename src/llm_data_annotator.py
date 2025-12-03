from utils import *
from llm_utils import *

import argparse
import pdb
import json
import copy
import csv
import re
import os
import glob
import pandas as pd
from tqdm import tqdm
from munch import Munch
from itertools import product
from collections import Counter
from data.data_utils import map_dataset_name_to_class


def load_batch_list(batch_list_file):
    """Load batch list from CSV file."""
    batches = []
    with open(batch_list_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            batches.append({
                'batch_id': row['batch_id'],
                'task_name': row['task_name'],
                'model': row['model'],
                'seed': int(row['seed']),
                'debug': row['debug'] == 'True',
                'results_folder': row['results_folder'],
                'prompt_description': row['prompt_description'],
                'temperature': float(row.get('temperature', 0.0)),
            })
    return batches


def backup_and_update_batch_list(batch_list_file, remaining_batches):
    """Backup old batch list and save updated one."""
    # Create backup
    timestamp = get_experiment_start_date()
    backup_file = batch_list_file.replace('.csv', f'_BACKUP_{timestamp}.csv')
    
    if os.path.exists(batch_list_file):
        import shutil
        shutil.copy2(batch_list_file, backup_file)
        print(f"Backed up batch list to: {backup_file}")
    else:
        print(f'Could not copy batch list file, as it does not exists: {batch_list_file}')
        pdb.set_trace()
    
    # Write remaining batches
    with open(batch_list_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['batch_id', 'task_name', 'model', 'seed', 'debug', 'results_folder', 'prompt_description', 'temperature'])
        for batch in remaining_batches:
            writer.writerow([
                batch['batch_id'],
                batch['task_name'],
                batch['model'], 
                batch['seed'],
                batch['debug'],
                batch['results_folder'],
                batch['prompt_description'],
                batch['temperature']
            ])


class LLMAnnotator:
    """Base class for LLM experiment replication with parameter grid search."""

    def __init__(self, config=None, debug=False, results_folder=None):
        """Initialize the experiment."""
        self.debug = debug
        if results_folder is None:
            self.results_folder = "annotations"
        else:
            self.results_folder = results_folder
        if debug:
            self.results_folder += "_DEBUG"


    def get_experiment_grid(self, config, models, seed, params):
        """Generate the experiment grid with all possible model-parameter combinations."""
        if seed is None:
            seed = 42
        if config is None:
            self.config =  Munch({'SEED': seed})
        else:
            self.config = config 
        if models is None or models == []:
            self.models = []
        else:
            self.models = models

        if self.debug:
            self.param_grid_setup = {
                'temperature': [0.0],
                'max_new_tokens': [10],
            }
        elif params is not None:
            all_params = params
        else:
            all_params = {
                'temperature': [0.0],
                'max_new_tokens': [10],
            }
        return all_params

    def setup_experiment(self, task_name, is_openai_batch_job):
        """Set up the experiment folder structure."""
        results_filepath = f"{self.results_folder}/{task_name}"
        os.makedirs(results_filepath, exist_ok=True)
        runtime_filepath = f"{results_filepath}/runtimes"

        experimet_start = get_experiment_start_date()
        results_file = f"{results_filepath}/{experimet_start}.csv"
        runtime_file = f"{runtime_filepath}/{experimet_start}.csv"

        if not is_openai_batch_job:
            with open(results_file, mode='w', newline='') as file:
                writer = csv.DictWriter(
                    file, fieldnames=[
                        'id',
                        'ground_truth',
                        'response_mapped',
                        'response',
                        'output_mapping_description',
                        'model',
                        'params',
                        'prompt_description',
                        'seed',
                        'temperature',
                    ])
                writer.writeheader()

            os.makedirs(runtime_filepath, exist_ok=True)

            # Create a new CSV file for runtime metrics
            with open(runtime_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'runtime',
                    'model',
                    'params',
                    'dataset_length',
                    'prompt_description',
                    'nr_of_compatible_output_mappings',
                    ])

        return results_file, runtime_file

    def format_prompt(self, prompt_text, text):
        # check if of type list
        formatted_prompt = copy.deepcopy(prompt_text)
        if isinstance(formatted_prompt, list):
            formatted_prompt[-1]['content'] = formatted_prompt[-1]['content'].format(
                text=text)
        else:
            formatted_prompt = formatted_prompt.format(text=text)
        return formatted_prompt

    def generate_param_grid(self, seed):
        """Generate all permutations of parameters with unique seeds."""
        # Merge base params with dataset-specific params
        combined_params = self.param_grid_setup.copy()
        combined_params.update(self.all_params)

        # Generate parameter grid
        param_list = [
            dict(zip(combined_params.keys(), values))
            for values in product(*combined_params.values())
        ]
        
        # Add incrementing seed to each parameter set
        for i, params in enumerate(param_list):
            params['seed'] = seed + i
        
        return param_list

    def load_dataset(self):
        """Load the dataset for experiments (to be implemented by subclasses)."""
        raise NotImplementedError(
            "Subclasses must implement load_dataset method")



    def experiment_already_done(self, runtime_file, model, params):
        """Check if the experiment has already been done."""
        # Extract the base pattern from the runtime_file path
        base_dir = os.path.dirname(os.path.dirname(runtime_file))  # Go up two levels
        filename = os.path.basename(runtime_file)
        # Create the glob pattern to match all instances of this file in any timestamp directory
        pattern = os.path.join(base_dir, "*", filename)
        # Find all matching files
        all_matching_runtime_files = glob.glob(pattern)
        for fn in all_matching_runtime_files:
            with open(fn, mode='r', newline='') as file:
                reader = csv.reader(file)
                for row in reader:
                    if row[1] == model and row[2] == str(params):
                        return True
        return False


    def check_completed_experiments(self, config=None, models=None, seed=None, params=None):
        """Check which experiments have already been completed."""

        completed_experiments = []
        not_completed_experiments = []

        if seed is None:
            seed = 42
        all_params = self.get_experiment_grid(config, models, seed, params)
        all_params.update(params)
        task_name = self.get_dataset_name()
        results_file, runtime_file = self.setup_experiment(task_name)
        dataset = self.load_dataset()
        if self.debug:
            dataset = dataset[:5]
        param_grid = self.generate_param_grid(seed)

        for model in tqdm(self.models, desc="Processing models", leave=True, total=len(self.models)):

            for params in tqdm(param_grid, desc=f"  Processing param grid for {model}", leave=True, total=len(param_grid)):
                prompt_details = params['prompt_details']
                prompt_description = prompt_details['description']
                prompt_text = prompt_details['prompt_text']

                # if 'few-shot' in prompt_details or ' cot' in prompt_details:
                #     continue
                if '[ban' not in prompt_details:
                    continue

                experiment_data = {
                    'model': model,
                    'params': params,
                    'runtime_file': runtime_file,
                    }
                if self.experiment_already_done(runtime_file, model, params):
                    completed_experiments.append(experiment_data)
                else:
                    not_completed_experiments.append(experiment_data)
        
        return completed_experiments, not_completed_experiments



    def setup_batch_metadata_file(self, task_name):
        """Set up batch metadata CSV file."""
        results_filepath = f"{self.results_folder}/{task_name}"
        batch_metadata_file = f"{results_filepath}/batch_metadata.csv"
        
        if not os.path.exists(batch_metadata_file):
            with open(batch_metadata_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'batch_id',
                    'model', 
                    'prompt_description',
                    'params',
                    'status',
                    'input_file_id',
                    'output_file_id',
                    'created_at',
                    'completed_at',
                    'request_count',
                    'input_file_path',
                    'output_file_path'
                ])
        
        return batch_metadata_file

    def update_batch_metadata(self, metadata_file, batch_data):
        """Update batch metadata CSV with new batch information."""
        with open(metadata_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                batch_data.get('batch_id', ''),
                batch_data.get('model', ''),
                batch_data.get('prompt_description', ''),
                batch_data.get('params', ''),
                batch_data.get('status', ''),
                batch_data.get('input_file_id', ''),
                batch_data.get('output_file_id', ''),
                batch_data.get('created_at', ''),
                batch_data.get('completed_at', ''),
                batch_data.get('request_count', ''),
                batch_data.get('input_file_path', ''),
                batch_data.get('output_file_path', '')
            ])


    def process_retrieved_batch_results(self, task_name, models, data_loader, batch_id, seed=42, params=None, temperature=0.0, prompt_description=None):
        """Process already-retrieved batch results into CSV format."""
        
        if params is None:
            decoding_params = {
                'temperature': temperature,
                'max_new_tokens': 20,
            }
        else:
            decoding_params = params
            decoding_params['temperature'] = temperature
        
        # Find the retrieved results file
        results_filepath = f"{self.results_folder}/{task_name}"
        batch_output_dir = f"{results_filepath}/batch_outputs"
        batch_output_path = f"{batch_output_dir}/retrieved_{batch_id}_output.jsonl"
        
        if not os.path.exists(batch_output_path):
            print(f"Error: Batch output file not found at {batch_output_path}")
            print("Run retrieve_results mode first to download the batch results.")
            return
        
        # Load the original dataset and prompts to reconstruct the mapping
        dataset = data_loader.load_dataset()
        if self.debug:
            dataset = dataset[:5]
        prompts = data_loader.get_prompts()
        
        # Create ID->datapoint lookup dictionary
        datapoint_lookup = {str(row['id']): row for _, row in dataset.iterrows()}
        
        # Get output mappings
        output_mapping = data_loader.get_all_output_mappings()
        
        # Filter prompts once if prompt_description is specified
        if prompt_description is not None:
            prompts = [p for p in prompts if p['description'] == prompt_description]
        
        if len(prompts) != 1:
            print(f'Not exactly one prompt. len(prompts)={len(prompts)}')
            pdb.set_trace()
        
        # Pre-compute compatible output mappings for each prompt
        prompt_mappings = []
        for p in prompts:
            compatible_mappings = {k: v for k, v in output_mapping.items() 
                                if ('compatible_output_mapping' not in p or 
                                    k in p.get('compatible_output_mapping', []))}
            prompt_mappings.append((p, compatible_mappings))
        
        # Initialize chatbot to get process_batch_response method
        chatbot = self.initialize_chatbot(models[0])
        
        # Setup results file
        results_file, runtime_file = self.setup_experiment(task_name, is_openai_batch_job=True)
        
        print(f"Processing batch results from: {batch_output_path}")
        
        # Count lines efficiently for progress bar
        with open(batch_output_path, 'r') as f:
            total_lines = sum(1 for _ in f)
        
        # Process each line in the batch output file
        results = []
        nr_of_failed_lines_parsing = 0
        nr_of_failed_lines_choices = 0
        with open(batch_output_path, 'r') as f:
            for line in tqdm(f, desc="Processing batch outputs", leave=True, total=total_lines):
                line = line.strip()
                if not line:
                    continue
                    
                # Parse the batch response
                batch_response_data = chatbot.process_batch_response(line)
                if batch_response_data is None:
                    continue
                
                # Parse the full response line to get custom_id
                response_line = json.loads(line)
                custom_id = response_line['custom_id']
                
                # Extract datapoint_id from custom_id (format: "req_{index}_{datapoint_id}")
                try:
                    # Remove "req_x_" prefix to get the actual datapoint_id
                    datapoint_id = re.sub(r'^req_\d+_', '', custom_id)
                except Exception as e:
                    print(f"Exception: {e}")
                    print(f"Warning: Could not parse custom_id: {custom_id}")
                    nr_of_failed_lines_parsing += 1
                    continue

                datapoint = datapoint_lookup.get(datapoint_id)
                
                if datapoint is None:
                    print(f"Warning: Could not find datapoint with id: {datapoint_id}")
                    nr_of_failed_lines_parsing += 1
                    continue
                
                try:
                    response = batch_response_data['choices'][0]['message']['content']
                except Exception as e:
                    print(f"Exception: {e}")
                    print(f"batch_response_data: {batch_response_data}")
                    nr_of_failed_lines_choices += 1
                    # pdb.set_trace()
                    continue

                # Use pre-filtered prompts and pre-computed mappings
                for prompt, compatible_mappings in prompt_mappings:
                    current_prompt_description = prompt['description']
                    
                    # Process with all appropriate output mappings
                    for mapping_description, mapping in compatible_mappings.items():
                        mapping_params = mapping.get('params', {}).get('mapping_options', {})
                        response_mapped = mapping['function'](response, mapping_params)
                        
                        results.append({
                            'id': datapoint['id'],
                            'ground_truth': datapoint['ground_truth'],
                            'response_mapped': response_mapped,
                            'response': response,
                            'output_mapping_description': mapping_description,
                            'model': models[0],
                            'params': str(decoding_params),
                            'prompt_description': current_prompt_description,
                            'seed': seed,
                            'temperature': temperature,
                        })

        # Check if file exists and is empty
        write_header = not os.path.exists(results_file) or os.path.getsize(results_file) == 0

        # Save results to CSV
        with open(results_file, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=[
                'id',
                'ground_truth',
                'response_mapped',
                'response',
                'output_mapping_description',
                'model',
                'params',
                'prompt_description',
                'seed',
                'temperature',
            ])
            if write_header:
                writer.writeheader()
            writer.writerows(results)
        
        print(f"Processed {len(results)} results and saved to: {results_file}")
        if nr_of_failed_lines_parsing > 0:
            print(f"~~~WARNING: {nr_of_failed_lines_parsing} of results could not be matched with a datapoint id!!!!")
        if nr_of_failed_lines_choices > 0:
            print(f"~~~WARNING: {nr_of_failed_lines_choices} of results did not contain an answer!!!!")
            


    def handle_batch_operations(self, task_name, models, data_loader, batch_mode, batch_id=None, batch_timeout=86400, seed=42, params=None, batch_list_file=None, temperature=0.0, prompt_description=None):
        """Handle batch operations that don't require full experiment setup."""
        
        if batch_mode == 'check_status':
            if batch_id is None:
                print("Error: --batch_id is required for check_status mode")
                pdb.set_trace()
                return
            
            # Initialize chatbot for the first model (assuming OpenAI)
            chatbot = self.initialize_chatbot(models[0])
            if not any(m_name in models[0] for m_name in ["gpt", "o1", "o3", "o4"]):
                print("Error: check_status mode only works with OpenAI models")
                pdb.set_trace()
                return
            
            batch_status = chatbot.check_batch_status(batch_id)
            if batch_status:
                print(f"Batch {batch_id} status: {batch_status.status}")
                print(f"Created at: {batch_status.created_at}")
                print(f"Request counts: {batch_status.request_counts}")
                if batch_status.completed_at:
                    print(f"Completed at: {batch_status.completed_at}")
            return
        
        elif batch_mode == 'retrieve_results':
            if batch_id is None:
                print("Error: --batch_id is required for retrieve_results mode")
                pdb.set_trace()
                return
            
            # Initialize chatbot for the first model (assuming OpenAI)
            chatbot = self.initialize_chatbot(models[0])
            if not any(m_name in models[0] for m_name in ["gpt", "o1", "o3", "o4"]):
                print("Error: retrieve_results mode only works with OpenAI models")
                pdb.set_trace()
                return
            
            # Create output path
            results_filepath = f"{self.results_folder}/{task_name}"
            batch_output_dir = f"{results_filepath}/batch_outputs"
            batch_output_path = f"{batch_output_dir}/retrieved_{batch_id}_output.jsonl"
            
            output_file = chatbot.retrieve_batch_results(batch_id, batch_output_path)
            if output_file:
                print(f"Batch results downloaded to: {output_file}")
            return
        
        elif batch_mode == 'process_results':
            if batch_id is None:
                print("Error: --batch_id is required for process_results mode")
                pdb.set_trace()
                return
            
            self.process_retrieved_batch_results(task_name, models, data_loader, batch_id, seed, params, temperature, prompt_description)
            return



    def save_batch_to_list(self, results_folder, batch_data, batch_list_file=None):
        """Save batch information to the batch list CSV file."""
        if batch_list_file is None:
            # Create new timestamped file if not provided
            timestamp = get_experiment_start_date()
            batch_list_file = f"{results_folder}/batch_list_{timestamp}.csv"
        
        # Create the batch list file if it doesn't exist
        if not os.path.exists(batch_list_file):
            with open(batch_list_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['batch_id', 'task_name', 'model', 'seed', 'debug', 'results_folder', 'prompt_description', 'temperature'])
        
        # Append batch info
        with open(batch_list_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                batch_data['batch_id'],
                batch_data['task_name'], 
                batch_data['model'],
                batch_data['seed'],
                batch_data['debug'],
                batch_data['results_folder'],
                batch_data['prompt_description'],
                batch_data['temperature']
            ])


    def load_missing_ids(self, missing_ids_file):
        """Load missing IDs from CSV file."""
        if not missing_ids_file:
            return {}
        
        missing_ids_df = pd.read_csv(missing_ids_file)
        missing_ids_dict = {}
        
        for _, row in missing_ids_df.iterrows():
            key = (row['task'], row['model'], row['prompt'], row['temperature'], row['seed'])
            if key not in missing_ids_dict:
                missing_ids_dict[key] = set()
            missing_ids_dict[key].add(row['missing_id'])
        
        return missing_ids_dict

    def load_missing_experiments(self, missing_experiments_file):
        """Load missing experiments from CSV file."""
        if not missing_experiments_file:
            return set()
        
        missing_experiments_df = pd.read_csv(missing_experiments_file)
        missing_experiments_set = set()
        
        for _, row in missing_experiments_df.iterrows():
            key = (row['task'], row['model'], row['prompt'], row['temperature'], row['seed'])
            missing_experiments_set.add(key)
        
        return missing_experiments_set


    def should_process_experiment(self, task_name, model, prompt_description, temperature, seed, missing_experiments_set):
        """Check if experiment should be processed based on missing experiments."""
        if not missing_experiments_set:
            return True
        
        key = (task_name, model, prompt_description, temperature, seed)
        return key in missing_experiments_set


    def should_process_datapoint(self, task_name, model, prompt_description, temperature, seed, datapoint_id, missing_ids_dict):
        """Check if datapoint should be processed based on missing IDs."""
        if not missing_ids_dict:
            return True
        
        key = (task_name, model, prompt_description, temperature, seed)
        if key not in missing_ids_dict:
            return False
        
        return datapoint_id in missing_ids_dict[key]


    def run_experiments(self, task_name, models, data_loader, config=None, seed=None, params=None, skip_if_already_done=True, use_batch_api=False, batch_mode='submit_and_wait', batch_timeout=86400, batch_id=None, batch_list_file=None, temperature=0.0, prompt_description=None, missing_ids_file=None, missing_experiments_file=None):
        """Run experiments on all models with the parameter grid and track runtime metrics."""
        
        # Load missing IDs and experiments if provided
        missing_ids_dict = self.load_missing_ids(missing_ids_file)
        missing_experiments_set = self.load_missing_experiments(missing_experiments_file)
        
        if seed is None:
            seed = 42
        if config is None:
            self.config = Munch({'SEED': seed})

        decoding_params = {
            'temperature': temperature,
            'max_new_tokens': 20,
        }
        if params is not None:
            decoding_params.update(params)

        # Handle batch operations that don't require full experiment setup
        if use_batch_api and batch_mode in ['check_status', 'retrieve_results', 'process_results']:
            self.handle_batch_operations(task_name, models, data_loader, batch_mode, batch_id, batch_timeout, seed, params, batch_list_file, temperature, prompt_description)
            return

        results_file, runtime_file = self.setup_experiment(task_name, is_openai_batch_job=use_batch_api)
        cnt = 0

        dataset = data_loader.load_dataset()
        if self.debug:
            dataset = dataset[:5]
        prompts = data_loader.get_prompts()

        # Get results_filepath for batch processing
        results_filepath = f"{self.results_folder}/{task_name}"

        # For submit modes, create timestamped batch list file ONCE per session (moved outside loops)
        batch_list_file_for_session = None
        if use_batch_api and batch_mode in ['submit', 'submit_and_wait']:
            timestamp = get_experiment_start_date()
            batch_list_file_for_session = f"{self.results_folder}/batch_list_{timestamp}.csv"
            print(f"Batch list will be saved to: {batch_list_file_for_session}")

        for model in tqdm(models, desc="Processing models", leave=True, total=len(models)):
            chatbot = self.initialize_chatbot(model)
            if any(m_name in model for m_name in ["gpt", "o1", "o3", "o4"]):
                is_openai_model = True
            else:
                is_openai_model = False
            
            # Setup batch metadata file if using batch API
            if use_batch_api and is_openai_model:
                batch_metadata_file = self.setup_batch_metadata_file(task_name)
            
            for p in tqdm(prompts, desc="Processing prompts", leave=True, total=len(prompts)):
                prompt_description = p['description']

                # Check if this experiment should be processed
                if not self.should_process_experiment(task_name, model, prompt_description, temperature, seed, missing_experiments_set):
                    print(f'\n~~Skipping experiment as it is not in missing experiments list:~~')
                    print(f'\ttask_name: {task_name}')
                    print(f'\tmodel: {model}')
                    print(f'\tprompt_description: {prompt_description}')
                    print(f'\ttemperature: {temperature}')
                    print(f'\tseed: {seed}\n')
                    continue

                output_mapping = data_loader.get_all_output_mappings()
                start_time = time.perf_counter()

                compatible_output_mappings = {k: v for k, v in output_mapping.items() if ('compatible_output_mapping' not in p or (
                    'compatible_output_mapping' in p and k in p.get('compatible_output_mapping', [])))}

                # Batch API processing for OpenAI models
                if use_batch_api and is_openai_model:
                    # Prepare all batch requests
                    batch_requests = []
                    request_mapping = {}  # Maps custom_id to (datapoint, mapping info)
                    
                    for index, datapoint in dataset.iterrows():
                        prompt = data_loader.format_prompt(p['prompt_text'], datapoint)
                        custom_id = f"req_{index}_{datapoint['id']}"
                        
                        batch_request = chatbot.prepare_batch_request(
                            custom_id=custom_id,
                            prompt=prompt,
                            **decoding_params,
                            seed=seed
                        )
                        batch_requests.append(batch_request)
                        request_mapping[custom_id] = (datapoint, compatible_output_mappings)
                    
                    # Create batch file
                    experimet_start = get_experiment_start_date()
                    batch_input_dir = f"{results_filepath}/batch_inputs"
                    batch_input_path = f"{batch_input_dir}/{experimet_start}_{model.replace('/', '_')}_{prompt_description.replace(' ', '_')[:12]}.jsonl"
                    
                    chatbot.create_batch_file(batch_requests, batch_input_path)

                    # Submit batch
                    batch_response = chatbot.submit_batch(batch_input_path)
                    if batch_response is None:
                        continue
                    
                    # Update metadata
                    batch_data = {
                        'batch_id': batch_response.id,
                        'model': model,
                        'prompt_description': prompt_description,
                        'params': str(decoding_params),
                        'status': batch_response.status,
                        'input_file_id': batch_response.input_file_id,
                        'output_file_id': batch_response.output_file_id,
                        'created_at': batch_response.created_at,
                        'completed_at': batch_response.completed_at,
                        'request_count': len(batch_requests),
                        'input_file_path': batch_input_path,
                        'output_file_path': ''
                    }
                    self.update_batch_metadata(batch_metadata_file, batch_data)
                    
                    # Save to batch list for later processing (removed the timestamp creation from here)
                    batch_list_data = {
                        'batch_id': batch_response.id,
                        'task_name': task_name,
                        'model': model,
                        'seed': seed,
                        'debug': self.debug,
                        'results_folder': self.results_folder,
                        'prompt_description': prompt_description,
                        'temperature': temperature
                    }
                    self.save_batch_to_list(self.results_folder, batch_list_data, batch_list_file_for_session)

                    if batch_mode == 'submit':
                        print(f"Batch submitted: {batch_response.id}")
                        continue
                    
                    # Wait for completion
                    completed_batch = chatbot.wait_for_batch_completion(batch_response.id, batch_timeout)
                    if completed_batch is None or completed_batch.status != "completed":
                        continue
                    
                    # Retrieve results
                    batch_output_dir = f"{results_filepath}/batch_outputs"
                    batch_output_path = f"{batch_output_dir}/{experimet_start}_{model.replace('/', '_')}_{prompt_description.replace(' ', '_')}_output.jsonl"
                    
                    output_file = chatbot.retrieve_batch_results(batch_response.id, batch_output_path)
                    if output_file is None:
                        continue
                    
                    # Process results
                    results = []
                    with open(output_file, 'r') as f:
                        for line in f:
                            batch_response_data = chatbot.process_batch_response(line.strip())
                            if batch_response_data is None:
                                continue
                            
                            response_line = json.loads(line.strip())
                            custom_id = response_line['custom_id']
                            datapoint, mappings = request_mapping[custom_id]
                            
                            # Extract response text (similar to sync processing)
                            response = batch_response_data['choices'][0]['message']['content']
                            
                            # Process with all appropriate output mappings
                            for mapping_description, mapping in mappings.items():
                                mapping_params = mapping.get('params', {}).get('mapping_options', {})
                                response_mapped = mapping['function'](response, mapping_params)
                                
                                results.append({
                                    'id': datapoint['id'],
                                    'ground_truth': datapoint['ground_truth'],
                                    'response_mapped': response_mapped,
                                    'response': response,
                                    'output_mapping_description': mapping_description,
                                    'model': model,
                                    'params': str(decoding_params),
                                    'prompt_description': prompt_description,
                                    'seed': seed,
                                    'temperature': temperature,
                                })
                    
                    # Save batch results to CSV
                    with open(results_file, mode='a', newline='') as file:
                        writer = csv.DictWriter(file, fieldnames=[
                            'id',
                            'ground_truth',
                            'response_mapped',
                            'response',
                            'output_mapping_description',
                            'model',
                            'params',
                            'prompt_description',
                            'seed',
                            'temperature',
                        ])
                        writer.writerows(results)
                    
                    # Calculate runtime
                    runtime = time.perf_counter() - start_time
                    
                    # Save runtime metrics to CSV
                    with open(runtime_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            runtime,
                            model,
                            str(decoding_params),
                            len(dataset),
                            prompt_description,
                            len(compatible_output_mappings),
                        ])
                    
                    # Update metadata with completion info
                    batch_data['status'] = 'completed'
                    batch_data['completed_at'] = completed_batch.completed_at
                    batch_data['output_file_path'] = batch_output_path
                    self.update_batch_metadata(batch_metadata_file, batch_data)
                    
                else:
                    for index, datapoint in tqdm(dataset.iterrows(), desc=f"    Evaluating with {model}", leave=True, total=len(dataset)):
                        # Check if this datapoint should be processed (for missing IDs functionality)
                        if not self.should_process_datapoint(task_name, model, prompt_description, temperature, seed, datapoint['id'], missing_ids_dict):
                            continue

                        # Generate prompt and get response
                        prompt = data_loader.format_prompt(p['prompt_text'], datapoint)
                        if is_openai_model:
                            full_response_object, response = chatbot(prompt, decoding_params, seed=seed)
                            # Create full_response_object filename
                            base_path = os.path.dirname(results_file)
                            base_filename = os.path.basename(results_file)
                            # Create full_response_objects directory
                            full_response_dir = os.path.join(base_path, 'full_response_objects')
                            os.makedirs(full_response_dir, exist_ok=True)
                            # Generate the filename
                            full_response_object_filename = os.path.join(
                                full_response_dir,
                                base_filename.replace('.csv', f'_{cnt}_{datapoint["id"]}.json')
                            )
                            cnt += 1
                            # Write full_response_object to file
                            with open(full_response_object_filename, 'w') as file:
                                json.dump(full_response_object, file, ensure_ascii=False, indent=4)
                        else:
                            response = chatbot(prompt, decoding_params, seed=seed)

                        # Process with all appropriate output mappings
                        batch_results = []
                        for mapping_description, mapping in compatible_output_mappings.items():

                            mapping_params = mapping.get('params', {}).get('mapping_options', {})
                            response_mapped = mapping['function'](response, mapping_params)

                            batch_results.append({
                                'id': datapoint['id'],
                                'ground_truth': datapoint['ground_truth'],
                                'response_mapped': response_mapped,
                                'response': response,
                                'output_mapping_description': mapping_description,
                                'model': model,
                                'params': str(decoding_params),
                                'prompt_description': prompt_description,
                                'seed': seed,
                                'temperature': temperature,
                            })

                        # Write results immediately after processing each datapoint
                        with open(results_file, mode='a', newline='') as file:
                            writer = csv.DictWriter(file, fieldnames=[
                                'id',
                                'ground_truth',
                                'response_mapped',
                                'response',
                                'output_mapping_description',
                                'model',
                                'params',
                                'prompt_description',
                                'seed',
                                'temperature',
                            ])
                            writer.writerows(batch_results)

                    # Calculate runtime
                    runtime = time.perf_counter() - start_time

                    # Save runtime metrics to CSV
                    with open(runtime_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            runtime,
                            model,
                            str(decoding_params),
                            len(dataset),
                            prompt_description,
                            len(compatible_output_mappings),
                        ])

                print(f"\n\n   >> done evaluating {model}, freeing up GPU memory")
                free_gpu_memory([model, chatbot])

        print(f"Runtime metrics saved to {runtime_file}")

    def initialize_chatbot(self, model):
        """Initialize a chatbot for the given model."""
        # This should create a ChatBot instance with the appropriate model
        return ChatBot(model, self.config).chatbot

    def get_dataset_name(self):
        """Get the name of the dataset (to be implemented by subclasses)."""
        raise NotImplementedError(
            "Subclasses must implement get_dataset_name method")

    def get_text_from_datapoint(self, datapoint):
        """Extract the text to be classified from a datapoint (to be implemented by subclasses)."""
        raise NotImplementedError(
            "Subclasses must implement get_text_from_datapoint method")

    def get_ground_truth_from_datapoint(self, datapoint):
        """Extract the ground truth from a datapoint (to be implemented by subclasses)."""
        raise NotImplementedError(
            "Subclasses must implement get_ground_truth_from_datapoint method")



class LLMAnnotatorOpenAIBatch(LLMAnnotator):

    def __init__(self, config=None, debug=False, results_folder=None):
        """Initialize the experiment."""
        self.debug = debug
        if results_folder is None:
            self.results_folder = "annotations"
        else:
            self.results_folder = results_folder



if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run LLM experiments with different models and seed')
    parser.add_argument('--task_names', type=str,
                        help='Comma-separated list of tasks to run (e.g., "emotion,humor")')
    parser.add_argument('--models', type=str, default=None, 
                        help='Comma-separated list of models to use (e.g., "Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct")')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for the experiment (default: 42)')
    parser.add_argument('--debug', action='store_true', 
                        help='Run in debug mode with reduced parameter set')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature for text generation (default: 0.0)')
    parser.add_argument('--results_folder', type=str, default=None, 
                        help='Path to the folder where the results should be saved')
    parser.add_argument('--just_check_if_already_completed', action='store_true', 
                        help='Just check if the experiment has already been completed and output the completed/missing experiment runs')
    
    # Missing experiments functionality
    parser.add_argument('--missing_ids_file', type=str, default=None,
                        help='Path to CSV file containing missing IDs to process (Normal processing mode only)')
    parser.add_argument('--missing_experiments_file', type=str, default=None,
                        help='Path to CSV file containing missing experiments to process (Normal processing mode only)')
    
    # Batch processing args
    parser.add_argument('--use_batch_api', action='store_true', 
                        help='Use OpenAI Batch API for processing requests')
    parser.add_argument('--batch_mode', type=str, choices=['submit', 'check_status', 'retrieve_results', 'submit_and_wait', 'process_results', 'check_retrieve_and_process'],
                        default='submit', help='Batch processing mode')
    parser.add_argument('--batch_id', type=str, default=None, 
                        help='Batch ID for checking status or retrieving results')
    parser.add_argument('--batch_timeout', type=int, default=86400, 
                        help='Maximum wait time for batch completion in seconds (default: 24 hours)')
    parser.add_argument('--batch_list_file', type=str, default=None,
                        help='Path to batch list file for check_retrieve_and_process mode')

    # Parse arguments
    args = parser.parse_args()
    
    # Validate missing experiments arguments
    if args.missing_ids_file and args.missing_experiments_file:
        print("Error: --missing_ids_file and --missing_experiments_file cannot be used simultaneously")
        exit(1)
    
    if (args.missing_ids_file or args.missing_experiments_file) and args.use_batch_api:
        print("Error: Missing experiments functionality is only available in Normal processing mode (not with --use_batch_api)")
        exit(1)

    print(f"Running tasks: {args.task_names}")
    print(f"Models: {args.models}")
    
    # Process models argument if provided
    models_list = None
    if args.models:
        models_list = [model.strip() for model in args.models.split(',') if model.strip() != '']
    
    print(f"  Models: {models_list}")
    print(f"  Seed: {args.seed}")
    print(f"  Debug: {args.debug}")
    print(f"  Results folder: {args.results_folder}")
    
    if args.missing_ids_file:
        print(f"  Missing IDs file: {args.missing_ids_file}")
    if args.missing_experiments_file:
        print(f"  Missing experiments file: {args.missing_experiments_file}")

    # Handle check_retrieve_and_process mode separately
    if args.use_batch_api and args.batch_mode == 'check_retrieve_and_process':
        if args.batch_list_file is None:
            print("Error: --batch_list_file is required for check_retrieve_and_process mode")
            print("Specify the path to your batch list CSV file")
            exit(1)
        all_batch_list_files = args.batch_list_file.split(',')
        for my_batch_file in tqdm(all_batch_list_files, desc=f"Running batch files", leave=True, total=len(all_batch_list_files)):

            batches = load_batch_list(my_batch_file)
            if len(batches) == 0:
                print(f"0 batches remaining in {my_batch_file}")
            else:
                print(f'\n\n###~~~ RUNNING BATCH LIST FILE {my_batch_file}~~~###\n\n')
                remaining_batches = []
                
                print(f"Processing {len(batches)} batches from {my_batch_file}")
                
                for i, batch_info in enumerate(batches):
                    print(f"\n--- Processing batch {i+1}/{len(batches)}: {batch_info['batch_id']} ---")
                    
                    try:
                        # Load the data_loader for this task
                        data_loader = map_dataset_name_to_class(batch_info['task_name'])
                        
                        # Create experiment instance with the batch's original settings
                        batch_experiment = LLMAnnotatorOpenAIBatch(
                            debug=batch_info['debug'],
                            results_folder=batch_info['results_folder']
                        )
                        
                        # Step 1: Check status
                        batch_experiment.run_experiments(
                            task_name=batch_info['task_name'],
                            models=[batch_info['model']],
                            data_loader=data_loader,
                            seed=batch_info['seed'],
                            use_batch_api=True,
                            batch_mode='check_status',
                            batch_id=batch_info['batch_id'],
                            temperature=batch_info['temperature'],
                            prompt_description=batch_info['prompt_description']
                        )
                        
                        # Get actual status to decide next steps
                        chatbot = batch_experiment.initialize_chatbot(batch_info['model'])
                        batch_status = chatbot.check_batch_status(batch_info['batch_id'])
                        
                        if batch_status and batch_status.status == 'completed':
                            print(f"Batch {batch_info['batch_id']} is completed. Retrieving and processing...")
                            
                            # Step 2: Retrieve results
                            batch_experiment.run_experiments(
                                task_name=batch_info['task_name'],
                                models=[batch_info['model']],
                                data_loader=data_loader,
                                seed=batch_info['seed'],
                                use_batch_api=True,
                                batch_mode='retrieve_results',
                                batch_id=batch_info['batch_id'],
                                temperature=batch_info['temperature'],
                                prompt_description=batch_info['prompt_description']
                            )
                            
                            # Step 3: Process results
                            batch_experiment.run_experiments(
                                task_name=batch_info['task_name'],
                                models=[batch_info['model']],
                                data_loader=data_loader,
                                seed=batch_info['seed'],
                                use_batch_api=True,
                                batch_mode='process_results',
                                batch_id=batch_info['batch_id'],
                                temperature=batch_info['temperature'],
                                prompt_description=batch_info['prompt_description']
                            )
                            
                            print(f"Successfully processed batch {batch_info['batch_id']}")
                            
                        else:
                            print(f"Batch {batch_info['batch_id']} status: {batch_status.status if batch_status else 'unknown'}")
                            print("Adding back to list for later processing")
                            remaining_batches.append(batch_info)
                            
                    except Exception as e:
                        print(f"Error processing batch {batch_info['batch_id']}: {e}")
                        print("Adding back to list for later processing")
                        remaining_batches.append(batch_info)
                
                # Backup and update the batch list using temp_loader
                backup_and_update_batch_list(my_batch_file, remaining_batches)
                
                processed_count = len(batches) - len(remaining_batches)
                print(f"\n--- Summary ---")
                print(f"Processed: {processed_count} batches")
                print(f"Remaining: {len(remaining_batches)} batches")
        
    else:
        # Normal processing mode - create experiment instance with command line arguments
        experiment_instance = LLMAnnotator(debug=args.debug, results_folder=args.results_folder)
        
        all_tasks = args.task_names.split(',')
        for task_name in tqdm(all_tasks, desc=f"Running tasks", leave=True, total=len(all_tasks)):
            tn = task_name.strip()
            if tn != '':
                data_loader = map_dataset_name_to_class(tn)
                print(f"\n---Running experiment: {tn}---")
                experiment_instance.run_experiments(
                    task_name=task_name, 
                    models=models_list, 
                    data_loader=data_loader, 
                    seed=args.seed,
                    use_batch_api=args.use_batch_api,
                    batch_mode=args.batch_mode,
                    batch_timeout=args.batch_timeout,
                    batch_id=args.batch_id,
                    temperature=args.temperature,
                    missing_ids_file=args.missing_ids_file,
                    missing_experiments_file=args.missing_experiments_file,
                )
