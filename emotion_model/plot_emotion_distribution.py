from datasets import Dataset
import matplotlib.pyplot as plt

# 加载合并后的数据集
combined = Dataset.load_from_disk("combined_emotion_dataset")

# 提取标签
labels = [example['label'] for example in combined]

# 获取标签名称
label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# 绘图
plt.figure(figsize=(8, 5))
plt.hist(labels, bins=range(len(label_names) + 1), align='left', rwidth=0.8)
plt.xticks(range(len(label_names)), label_names, rotation=45)
plt.title("Emotion Label Distribution (Combined Dataset)")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# 保存图片
plt.savefig("combined_emotion_dist.png")

# 显示图片
plt.show()
