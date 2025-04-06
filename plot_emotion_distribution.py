from datasets import load_dataset
import matplotlib.pyplot as plt

# Load dataset
dataset = load_dataset("dair-ai/emotion")
train_dataset = dataset['train']

# Get label names and data
label_names = dataset['train'].features['label'].names
labels = [example['label'] for example in train_dataset]

# Plot and save
plt.figure(figsize=(8,5))
plt.hist(labels, bins=range(len(label_names)+1), align='left', rwidth=0.8)
plt.xticks(range(len(label_names)), label_names, rotation=45)
plt.title("Emotion Label Distribution (Training Set)")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 💾 Save in the same folder as your script
plt.savefig("emotion_dist.png")

# Optional: still show it
plt.show()
