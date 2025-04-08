from datasets import load_dataset, Dataset, concatenate_datasets

# Load both datasets
goemotions = load_dataset("go_emotions", split="train")
emotion_ds = load_dataset("dair-ai/emotion", split="train")


label_names = goemotions.features["labels"].feature.names
emotion_map = {
    "sadness": ["sadness", "grief", "remorse", "disappointment"],
    "joy": ["joy", "amusement", "excitement", "gratitude", "optimism", "relief", "admiration"],
    "love": ["love", "caring", "desire"],
    "anger": ["anger", "annoyance", "disapproval", "disgust"],
    "fear": ["fear", "nervousness"],
    "surprise": ["surprise", "realization", "curiosity"]
}


reverse_map = {}
for k, v in emotion_map.items():
    for label in v:
        reverse_map[label] = k

# Function: find first matching label and map to your category
def map_labels(example):
    new_labels = [reverse_map[label_names[i]] for i in example["labels"] if label_names[i] in reverse_map]
    example["text"] = example["text"]
    example["label"] = new_labels[0] if new_labels else None
    return example

# Apply mapping
goemotions = goemotions.map(map_labels)
goemotions = goemotions.filter(lambda x: x["label"] is not None)
label_to_id = {label: idx for idx, label in enumerate(["sadness", "joy", "love", "anger", "fear", "surprise"])}
goemotions = goemotions.map(lambda x: {"label": label_to_id[x["label"]]})

# Keep only relevant columns
goemotions = goemotions.remove_columns(set(goemotions.column_names) - {"text", "label"})

from datasets import Value, Features

emotion_ds = emotion_ds.cast(Features({'text': Value('string'), 'label': Value('int64')}))

combined = concatenate_datasets([emotion_ds, goemotions])


print(f"Combined dataset size: {len(combined)}")
print(combined[0])
print(goemotions.column_names)

# Save the combined dataset
combined.save_to_disk("combined_emotion_dataset")


