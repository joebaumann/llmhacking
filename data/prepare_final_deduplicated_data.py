import pdb
from tqdm import tqdm
from data.data_utils import process_full_dataset, all_tasks

for task_name in tqdm(all_tasks, desc='Processing tasks', total=len(all_tasks)):
    print(f'\n--processing task: {task_name}--')
    df_final = process_full_dataset(task_name)
