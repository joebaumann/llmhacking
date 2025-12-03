# run as python -m data.prompt_paraphrasing

import pdb
import pandas as pd
import os
import copy
import json
from tqdm import tqdm
from data.data_utils import map_dataset_name_to_class, all_tasks
from llm_utils import ChatBot
from utils import get_experiment_start_date

experiment_start = get_experiment_start_date()

paraphraser_model, params = 'o3-2025-04-16', {'reasoning_params': {
    'max_completion_tokens': 20000, 'reasoning_effort': "high"}}

paraphraser_prompt = """Paraphrase the following prompt that will be used for automated data annotation.

Key requirements:
- Preserve the exact same annotation task
- Keep all placeholders (e.g., {{text}}) unchanged
- Ensure the paraphrased prompt guides the model to output ONLY the keys from this mapping dictionary:
  {prompt_output_mapping_placeholder}
- Base your paraphrase on the content in the original prompt without adding new information
- Improve clarity and structure while maintaining the original meaning

Original prompt to paraphrase:
{prompt_placeholder}

Provide only the paraphrased prompt."""

paraphraser = ChatBot(paraphraser_model).chatbot

seed = 42

for task_name in tqdm(all_tasks, desc='Processing tasks', total=len(all_tasks)):
    print(f'doing task:', task_name)
    data_loader = map_dataset_name_to_class(task_name)
    # dataset = data_loader.load_dataset()
    prompts = data_loader.get_prompts()
    output_mappings = data_loader.get_all_output_mappings()
    nr_of_prompts = len(prompts)
    prompt_data = []
    for p in prompts:
        prompt_dict = {'task_name': task_name}
        prompt_dict.update(p)
        prompt_data.append(prompt_dict)
    prompt_data_df = pd.DataFrame(prompt_data)
    prompt_candidates = prompt_data_df[prompt_data_df['description'].str.contains(
        'zero-shot-short')]
    if len(prompt_candidates) == 0:
        prompt_candidates = prompt_data_df[prompt_data_df['description'].str.contains(
            'zero-shot')]
    if len(prompt_candidates) == 0:
        prompt_candidates = prompt_data_df[prompt_data_df['description'].str.contains(
            'original')]
    if len(prompt_candidates) == 0:
        prompt_candidates = prompt_data_df
    prompt_candidates['original'] = prompt_candidates['description'].str.contains(
        'original')
    prompt_candidates.sort_values('original', ascending=False, inplace=True)
    paraphrase_counter = 0
    while nr_of_prompts < 5:
        seed += 1
        paraphraser_prompt_text = copy.deepcopy(paraphraser_prompt)
        candidate_prompt = prompt_candidates.iloc[paraphrase_counter % len(
            prompt_candidates)]
        output_mapping_name = candidate_prompt['compatible_output_mapping'][0]
        prompt_output_mapping = output_mappings[output_mapping_name].get(
            'params', {}).get('mapping_options', {})
        paraphraser_prompt_text = paraphraser_prompt_text.format(
            prompt_output_mapping_placeholder=prompt_output_mapping,
            prompt_placeholder=candidate_prompt['prompt_text'],
        )
        pf = paraphraser(paraphraser_prompt_text, seed=seed, **params)
        fn = f'data/{data_loader.data_directory}/prompt_paraphrases'
        os.makedirs(fn, exist_ok=True)
        fn += f'/prompt_paraphrase_{task_name}_{paraphraser_model}_{paraphrase_counter}_{experiment_start}.json'

        # Create JSON data with both original prompt info and paraphrased result
        json_data = {
            'paraphrased_prompt': pf,
            'compatible_output_mapping': candidate_prompt['compatible_output_mapping'],
            'original_description': candidate_prompt['description'],
            'original_prompt_text': candidate_prompt['prompt_text'],
            'output_mapping': prompt_output_mapping,
            'paraphrase_counter': paraphrase_counter,
            'task_name': candidate_prompt['task_name'],
        }

        with open(fn, 'w') as f:
            json.dump(json_data, f, indent=2)

        paraphrase_counter += 1
        nr_of_prompts += 1
