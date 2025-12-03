from utils import *
from llm_utils import *

import argparse
import pdb
import json
import csv
import os
import pandas as pd
from tqdm import tqdm
from munch import Munch
from itertools import product
from collections import Counter
import time
import json
from datetime import datetime

from data.data_utils import map_dataset_name_to_class
from src.helpers import load_prepared_prompts, load_confidence_csv_file, load_confidence_results


class ConfidenceDataLoader():
    def __call__(self, task_name, model, filepath):
        model_name = model.replace('/', '--')
        outpath_task_model = Path(filepath, task_name, model_name, 'all_prompts_for_confidence_elicitation.csv')
        
        if not outpath_task_model.exists():
            raise FileNotFoundError(f"Could not find confidence data file: {outpath_task_model}")
        
        confidence_df = pd.read_csv(outpath_task_model)
        return confidence_df



class LLMConfidenceElicitator:

    def __init__(self, config=None, debug=False, results_folder=None, skip_processed=True):
        self.debug = debug
        self.skip_processed = skip_processed
        if results_folder is None:
            self.results_folder = "results_llm_confidence"
        else:
            self.results_folder = results_folder

    def setup_experiment(self, task_name, model):
        """Set up the experiment folder structure."""
        model_name = model.replace('/', '--')
        results_filepath = f"{self.results_folder}/{task_name}/{model_name}"
        os.makedirs(results_filepath, exist_ok=True)
        runtime_filepath = f"{results_filepath}/runtimes"

        experimet_start = get_experiment_start_date()
        results_file = f"{results_filepath}/{experimet_start}.csv"
        runtime_file = f"{runtime_filepath}/{experimet_start}.csv"

        with open(results_file, mode='w', newline='') as file:
            writer = csv.DictWriter(
                file, fieldnames=[
                    'id',
                    'response_mapped',
                    'confidence',
                    'model',
                    'params',
                    'seed',
                    'temperature',
                    # 'text',
                    # 'prompt'
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
                ])

        return results_file, runtime_file

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


    def run_batch_submit(self, task_name, models, config=None, seed=None, params=None, temperature=0.0, skip_processed=True):
        """Submit batch requests for all models."""
        self.skip_processed = skip_processed
        
        if seed is None:
            seed = 42
        if config is None:
            self.config = Munch({'SEED': seed})

        decoding_params = {
            'temperature': temperature,
            'max_new_tokens': 10,
        }
        if params is not None:
            decoding_params.update(params)

        for model in tqdm(models, desc="Submitting batch requests", leave=True, total=len(models)):
            if not any(m_name in model for m_name in ["gpt", "o1", "o3", "o4"]):
                print(f"Skipping {model} - not an OpenAI model")
                continue

            results_file, runtime_file = self.setup_experiment(task_name, model)
            
            confidence_loader = ConfidenceDataLoader()
            dataset = confidence_loader(task_name, model, self.results_folder)

            # Load already processed combinations
            if self.skip_processed:
                already_processed_df = load_confidence_results(self.results_folder, task_name, model)
                if not already_processed_df.empty:
                    already_processed_combinations = set(
                        zip(already_processed_df['id'], already_processed_df['response_mapped'])
                    )
                else:
                    already_processed_combinations = set()
            else:
                already_processed_combinations = set()

            if self.debug:
                dataset = dataset[:5]

            chatbot = self.initialize_chatbot(model)
            
            # Prepare batch requests
            batch_requests = []
            id_mapping = {}
            
            for index, datapoint in dataset.iterrows():
                if (datapoint['id'], datapoint['response_mapped']) in already_processed_combinations:
                    continue
                    
                prompt = datapoint['confidence_prompt'].replace(
                    'Output only a single number between 0 and 1.', 
                    'Output only a single number between 0 and 1, without any context or explanation.'
                )
                
                custom_id = f"{datapoint['id']}_{datapoint['response_mapped']}_{index}"
                id_mapping[custom_id] = {
                    'id': datapoint['id'],
                    'response_mapped': datapoint['response_mapped']
                }
                
                request = chatbot.prepare_batch_request(
                    custom_id=custom_id,
                    prompt=prompt,
                    **decoding_params,
                    seed=seed
                )

                request = chatbot.prepare_batch_request(
                    custom_id=custom_id,
                    prompt=prompt,
                    temperature=decoding_params['temperature'],
                    max_new_tokens=decoding_params['max_new_tokens'],
                    seed=seed
                )
                batch_requests.append(request)

            if not batch_requests:
                print(f"No new requests for {model}")
                continue

            # Create batch file
            model_name = model.replace('/', '--')
            batch_dir = f"{self.results_folder}/{task_name}/{model_name}/batch_files"
            timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            batch_file = f"{batch_dir}/{timestamp}_requests.jsonl"
            
            chatbot.create_batch_file(batch_requests, batch_file)
            
            # Submit batch
            batch_response = chatbot.submit_batch(batch_file)
            
            if batch_response:
                # Save ID mapping
                mapping_file = f"{batch_dir}/{timestamp}_id_mapping.json"
                with open(mapping_file, 'w') as f:
                    json.dump(id_mapping, f, indent=2)
                
                # Save batch info
                batch_info = {
                    'batch_id': batch_response.id,
                    'timestamp': timestamp,
                    'model': model,
                    'task_name': task_name,
                    'mapping_file': mapping_file,
                    'num_requests': len(batch_requests)
                }
                
                # Update submitted batches file
                batches_file = f"{self.results_folder}/{task_name}/{model_name}/submitted_batches.json"
                if os.path.exists(batches_file):
                    with open(batches_file, 'r') as f:
                        submitted_batches = json.load(f)
                else:
                    submitted_batches = []
                
                submitted_batches.append(batch_info)
                os.makedirs(os.path.dirname(batches_file), exist_ok=True)
                with open(batches_file, 'w') as f:
                    json.dump(submitted_batches, f, indent=2)
                
                print(f"Submitted batch {batch_response.id} for {model} with {len(batch_requests)} requests")

    def run_batch_check_retrieve_process(self, task_name, models, config=None, seed=None, params=None, temperature=0.0):
        """Check, retrieve and process batch results."""
        if seed is None:
            seed = 42
        if config is None:
            self.config = Munch({'SEED': seed})

        decoding_params = {
            'temperature': temperature,
            'max_new_tokens': 10,
        }
        if params is not None:
            decoding_params.update(params)

        for model in tqdm(models, desc="Processing batch results", leave=True, total=len(models)):
            if not any(m_name in model for m_name in ["gpt", "o1", "o3", "o4"]):
                continue

            model_name = model.replace('/', '--')
            batches_file = f"{self.results_folder}/{task_name}/{model_name}/submitted_batches.json"
            
            if not os.path.exists(batches_file):
                print(f"No submitted batches found for {model}")
                continue

            with open(batches_file, 'r') as f:
                submitted_batches = json.load(f)

            if not submitted_batches:
                continue

            chatbot = self.initialize_chatbot(model)
            processed_batches = []
            remaining_batches = []

            for batch_info in submitted_batches:
                batch_id = batch_info['batch_id']
                
                # Check batch status
                batch_status = chatbot.check_batch_status(batch_id)
                if not batch_status or batch_status.status != 'completed':
                    remaining_batches.append(batch_info)
                    print(f"Batch {batch_id} not ready (status: {batch_status.status if batch_status else 'unknown'})")
                    continue

                # Retrieve results
                batch_results_timestamp = get_experiment_start_date()
                results_dir = f"{self.results_folder}/{task_name}/{model_name}/batch_results"
                results_file = f"{results_dir}/{batch_results_timestamp}_results.jsonl"
                
                retrieved_file = chatbot.retrieve_batch_results(batch_id, results_file)
                if not retrieved_file:
                    remaining_batches.append(batch_info)
                    continue

                # Load ID mapping
                with open(batch_info['mapping_file'], 'r') as f:
                    id_mapping = json.load(f)

                # Process results
                results_csv = f"{self.results_folder}/{task_name}/{model_name}/{batch_results_timestamp}.csv"
                
                with open(results_csv, mode='w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=[
                        'id', 'response_mapped', 'confidence', 'model', 'params', 'seed', 'temperature'
                    ])
                    writer.writeheader()

                    with open(retrieved_file, 'r') as f:
                        for line in f:
                            response_data = chatbot.process_batch_response(line.strip())
                            if not response_data:
                                continue

                            custom_id = json.loads(line)['custom_id']
                            confidence = response_data['choices'][0]['message']['content']
                            
                            mapping = id_mapping[custom_id]
                            
                            writer.writerow({
                                'id': mapping['id'],
                                'response_mapped': mapping['response_mapped'],
                                'confidence': confidence,
                                'model': model,
                                'params': str(decoding_params),
                                'seed': seed,
                                'temperature': temperature,
                            })

                processed_batches.append(batch_info)
                print(f"Processed batch {batch_id} for {model}")

            # Update submitted batches file
            if processed_batches:
                # Backup current file
                backup_file = f"{batches_file}.backup_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
                if os.path.exists(batches_file):
                    os.rename(batches_file, backup_file)
                
                # Save remaining batches
                with open(batches_file, 'w') as f:
                    json.dump(remaining_batches, f, indent=2)


    def run_experiments(self, task_name, models, config=None, seed=None, params=None, temperature=0.0, skip_processed=True, batch_mode=None):
        """Run experiments on all models with the parameter grid and track runtime metrics."""
        
        if batch_mode == 'submit':
            return self.run_batch_submit(task_name, models, config, seed, params, temperature, skip_processed)
        elif batch_mode == 'check_retrieve_and_process':
            return self.run_batch_check_retrieve_process(task_name, models, config, seed, params, temperature)
        
        # Original sync processing code remains unchanged
        self.skip_processed = skip_processed

        if seed is None:
            seed = 42
        if config is None:
            self.config = Munch({'SEED': seed})

        decoding_params = {
            'temperature': temperature,
            'max_new_tokens': 10,
        }
        if params is not None:
            decoding_params.update(params)

        cnt = 0

        for model in tqdm(models, desc="Processing models", leave=True, total=len(models)):
            results_file, runtime_file = self.setup_experiment(task_name, model)

            confidence_loader = ConfidenceDataLoader()
            dataset = confidence_loader(task_name, model, self.results_folder)

            # load all already processed id-response_mapped-combinations
            if self.skip_processed:
                already_processed_df = load_confidence_results(self.results_folder, task_name, model)
                if not already_processed_df.empty:
                    already_processed_id_response_mapped_combinations = set(
                        zip(already_processed_df['id'], already_processed_df['response_mapped'])
                    )
                else:
                    already_processed_id_response_mapped_combinations = set()
            else:
                already_processed_id_response_mapped_combinations = set()

            
            if self.debug:
                dataset = dataset[:5]

            chatbot = self.initialize_chatbot(model)
            if any(m_name in model for m_name in ["gpt", "o1", "o3", "o4"]):
                is_openai_model = True
            else:
                is_openai_model = False
            
            start_time = time.perf_counter()

            for index, datapoint in tqdm(dataset.iterrows(), desc=f"    Evaluating with {model}", leave=True, total=len(dataset)):
                # skip already processed datapoints
                if (datapoint['id'], datapoint['response_mapped']) in already_processed_id_response_mapped_combinations:
                    continue

                # Generate prompt and get confidence
                prompt = datapoint['confidence_prompt'].replace('Output only a single number between 0 and 1.', 'Output only a single number between 0 and 1, without any context or explanation.')
                if is_openai_model:
                    full_response_object, confidence = chatbot(prompt, decoding_params, seed=seed)
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
                    confidence = chatbot(prompt, decoding_params, seed=seed)


                # Write results immediately after processing each datapoint
                with open(results_file, mode='a', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=[
                        'id',
                        'response_mapped',
                        'confidence',
                        'model',
                        'params',
                        'seed',
                        'temperature',
                    ])
                    writer.writerow({
                        'id': datapoint['id'],
                        'response_mapped': datapoint['response_mapped'],
                        'confidence': confidence,
                        'model': model,
                        'params': str(decoding_params),
                        'seed': seed,
                        'temperature': temperature,
                    })

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
                ])

            print(f"\n\n   >> done evaluating {model}, freeing up GPU memory")
            free_gpu_memory([model, chatbot])

        print(f"Runtime metrics saved to {runtime_file}")

    def initialize_chatbot(self, model):
        """Initialize a chatbot for the given model."""
        # This should create a ChatBot instance with the appropriate model
        return ChatBot(model, self.config).chatbot


if __name__ == "__main__":

    # Need to run `python -m src.llm_verbalized_confidence_data_preparation` first, to prepare the prompt for each task and instance

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
    parser.add_argument('--skip_processed', action='store_true', default=True,
                        help='Skip already processed datapoints (default: True)')
    parser.add_argument('--batch_mode', type=str, choices=['submit', 'check_retrieve_and_process'], default=None,
                        help='Run in batch mode: submit or check_retrieve_and_process')

    # Parse arguments
    args = parser.parse_args()
    
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
    
    # Normal processing mode - create experiment instance with command line arguments
    experiment_instance = LLMConfidenceElicitator(
        debug=args.debug, 
        results_folder=args.results_folder,
        skip_processed=args.skip_processed
    )
    all_tasks = args.task_names.split(',')
    for task_name in tqdm(all_tasks, desc=f"Running tasks", leave=True, total=len(all_tasks)):
        task_name = task_name.strip()
        if task_name != '':
            print(f"\n---Running experiment: {task_name}---")
            experiment_instance.run_experiments(
                task_name=task_name, 
                models=models_list, 
                seed=args.seed,
                temperature=args.temperature,
                skip_processed=args.skip_processed,
                batch_mode=args.batch_mode,
            )
