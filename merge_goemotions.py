from datasets import load_dataset, Dataset, concatenate_datasets

# Load both datasets
goemotions = load_dataset("go_emotions", split="train")
emotion_ds = load_dataset("dair-ai/emotion", split="train")

# Mapping from GoEmotions label index to label name
label_names = goemotions.features["labels"].feature.names

# Define mapping: which GoEmotion labels map to your 6 categories
emotion_map = {
    "sadness": ["sadness", "grief", "remorse", "disappointment"],
    "joy": ["joy", "amusement", "excitement", "gratitude", "optimism", "relief", "admiration"],
    "love": ["love", "caring", "desire"],
    "anger": ["anger", "annoyance", "disapproval", "disgust"],
    "fear": ["fear", "nervousness"],
    "surprise": ["surprise", "realization", "curiosity"]
}

# Reverse map: label_name → your category
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
# Remove examples without matching labels
goemotions = goemotions.filter(lambda x: x["label"] is not None)

# Make sure labels are numerical (like dair-ai/emotion)
label_to_id = {label: idx for idx, label in enumerate(["sadness", "joy", "love", "anger", "fear", "surprise"])}
goemotions = goemotions.map(lambda x: {"label": label_to_id[x["label"]]})

# Keep only relevant columns
goemotions = goemotions.remove_columns(set(goemotions.column_names) - {"text", "label"})
# Convert emotion_ds's label to plain int64 for alignment
from datasets import Value, Features

emotion_ds = emotion_ds.cast(Features({'text': Value('string'), 'label': Value('int64')}))

# Combine with dair-ai/emotion dataset
combined = concatenate_datasets([emotion_ds, goemotions])

# Check size
print(f"Combined dataset size: {len(combined)}")
print(combined[0])
print(goemotions.column_names)

