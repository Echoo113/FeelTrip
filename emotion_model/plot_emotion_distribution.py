from datasets import Dataset
import matplotlib.pyplot as plt


combined = Dataset.load_from_disk("emotion_model/combined_emotion_dataset")


labels = [example['label'] for example in combined]


label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# plot the distribution of labels
plt.figure(figsize=(8, 5))
plt.hist(labels, bins=range(len(label_names) + 1), align='left', rwidth=0.8)
plt.xticks(range(len(label_names)), label_names, rotation=45)
plt.title("Emotion Label Distribution ")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()


plt.savefig("combined_emotion_dist.png")


plt.show()
