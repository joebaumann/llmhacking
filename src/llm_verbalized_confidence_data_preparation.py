# run as: python -m src.llm_verbalized_confidence_data_preparation
import pdb
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import ast
# import os
from data.data_utils import map_dataset_name_to_class, all_tasks
from src.helpers import load_llm_annotated_data


def get_prompts_for_all_annotations_by_all_model(datasets_llm_annotated, data_loader):
    cols_to_keep = ['id', 'response_mapped', 'model']
    dataset = data_loader.load_dataset()
    confidence_elicitation_prompt_template = data_loader.get_prompt_for_verbalized_confidence_elicitation()

    # drop all rows where response_mapped=='na'
    # get all annotations for each text (by id) for each model datasets_llm_annotated, i.e., I want one row for each unique combination of 'id', 'response_mapped', 'model'
    annotations_by_model = datasets_llm_annotated[datasets_llm_annotated['response_mapped'] != 'na'].drop_duplicates(
        cols_to_keep)

    model_nan_df = annotations_by_model[annotations_by_model['model'].isna()]
    if len(model_nan_df) > 0:
        pdb.set_trace()

    annotations_by_model = annotations_by_model[cols_to_keep]

    # first ensure that the dataset df does not contain the same column names
    for c in cols_to_keep:
        if c in dataset.columns and c != 'id':
            dataset.rename(columns={c: f'{c}_merged'}, inplace=True)
    # merge left with dataset on id
    annotations_by_model = annotations_by_model.merge(
        dataset, on='id', how='left')
    annotations_by_model['confidence_prompt'] = annotations_by_model.apply(lambda row: data_loader.format_confidence_elicitation_prompt(
        confidence_elicitation_prompt_template, row, row['response_mapped']), axis=1)

    try:
        annotations_by_model = annotations_by_model[cols_to_keep + [
            'confidence_prompt']]
    except Exception as e:
        print(e)
        pdb.set_trace()

    return annotations_by_model


def save_prompts_to_disk_for_llm_confidence_elicitation(outpath, task_name, prompts_for_all_annotations_by_all_model):
    if prompts_for_all_annotations_by_all_model.empty or 'model' not in prompts_for_all_annotations_by_all_model.columns:
        pdb.set_trace()
    all_models = prompts_for_all_annotations_by_all_model['model'].unique()
    if all_models is None or len(all_models) == 0:
        pdb.set_trace()
    for model in all_models:
        model_name = model.replace('/', '--')
        outpath_task_model = Path(outpath, task_name, model_name)
        # create dir if not exists
        outpath_task_model.mkdir(parents=True, exist_ok=True)
        filename = Path(outpath_task_model,
                        'all_prompts_for_confidence_elicitation.csv')

        # save prompts_for_all_annotations_by_all_model to disk with filename
        model_data = prompts_for_all_annotations_by_all_model[
            prompts_for_all_annotations_by_all_model['model'] == model]
        model_data.to_csv(filename, index=False)


def main(outpath):
    for task_name in tqdm(all_tasks, desc='Processing tasks', leave=True, total=len(all_tasks)):
        data_loader = map_dataset_name_to_class(task_name)
        # load all llm annotated data
        datasets_llm_annotated = load_llm_annotated_data(task_name)
        prompts_for_all_annotations_by_all_model = get_prompts_for_all_annotations_by_all_model(
            datasets_llm_annotated, data_loader)

        save_prompts_to_disk_for_llm_confidence_elicitation(
            outpath, task_name, prompts_for_all_annotations_by_all_model)

        print(
            f'Finished task {task_name}. Results saved to {Path(outpath, task_name)}')
    print(
        f'\n~~Saved prompts for llm confidence elicitation to disk for all tasks. Results saved to {outpath}')


if __name__ == "__main__":
    outpath = 'results_llm_confidence'
    main(outpath)
