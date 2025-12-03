import pandas as pd
import json
import numpy as np
import krippendorff


all_json_files = [
    ('ChatGPT', 'data/factuality/data_raw/data/labeled/ChatGPT.jsonl'),
    ('InstructGPT', 'data/factuality/data_raw/data/labeled/InstructGPT.jsonl'),
    ('PerplexityAI', 'data/factuality/data_raw/data/labeled/PerplexityAI.jsonl'),
]


topics, generations, atomic_facts, ground_truths, models = [], [], [], [], []
for model_name, input_path in all_json_files:
    with open(input_path) as f:
        for line in f:
            dp = json.loads(line)
            if dp["annotations"] is not None:
                for sent in dp["annotations"]:
                    if sent["human-atomic-facts"] is not None:
                        for atom in sent["human-atomic-facts"]:
                            topics.append(dp["topic"])
                            generations.append(dp["output"])
                            atomic_facts.append(atom["text"])
                            ground_truths.append(atom["label"])
                            models.append(model_name)


df = pd.DataFrame({
    'topic': topics,
    'generation': generations,
    'atomic_fact': atomic_facts,
    'ground_truth': ground_truths,
    'model': models
})


df['ground_truth'] = df['ground_truth'].replace({
    'S': 'Supported',
    'NS': 'Not-supported',
    'IR': 'Irrelevant',
})

df.rename(columns={'atomic_fact': 'text'}, inplace=True)

df['topic'] = df['topic'].str.strip()
df['generation'] = df['generation'].str.strip()


print(f"\nFinal dataset shape: {df.shape}")
print(f"Final columns: {list(df.columns)}")
print(
    f"Ground truth distribution: {df['ground_truth'].value_counts().sort_index()}")

df.reset_index(drop=True, inplace=True)

# sort df columns to start with 'ground_truth', then 'text', and then all other columns
priority_columns = ['ground_truth', 'text']
other_columns = [
    col for col in df.columns if col not in priority_columns]
df = df[priority_columns + other_columns]

df.to_csv('data/all_data_processed_full/factuality.csv', index=False)


### estimate annotator agreement ###

# get number of samples used for reported disagreement
n = int(len(df) * 0.1)
reported_disagreement = 1-0.91
# use label distribution from full df
labels = [0, 1, 2]
p = df['ground_truth'].value_counts(dropna=False).values/len(df)
alphas = []

for seed in range(10000):
    rng = np.random.RandomState(seed)
    a1 = rng.choice(labels, n, p=p)
    a2 = a1.copy()
    flip_idx = rng.choice(n, int(n * reported_disagreement), replace=False)
    for i in flip_idx:
        a2[i] = rng.choice([l for l in labels if l != a2[i]])
    alphas.append(krippendorff.alpha([a1, a2], level_of_measurement='nominal'))

print(f"Average Krippendorff's alpha: {np.mean(alphas):.4f} (±{np.std(alphas):.4f})")
