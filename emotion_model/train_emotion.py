import torch
import pandas as pd
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 加载 arrow 格式数据集
dataset = load_from_disk("/Users/echohe/Desktop/FeelTrip/emotion_model/combined_emotion_dataset")

dataset = dataset["train"] if "train" in dataset else dataset

# 2. 转成 pandas DataFrame
df = dataset.to_pandas()
print("Column names:", df.columns)
print(df.head())

# 3. 提取文本和标签
texts = df["text"].tolist()
labels = df["label"].tolist()

# 4. 标签数
num_labels = len(set(labels))

# 5. 划分训练/验证集
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# 6. 初始化分词器
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# 7. 创建数据集类
class EmotionDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 8. 构建 DataLoader
train_dataset = EmotionDataset(train_texts, train_labels)
val_dataset = EmotionDataset(val_texts, val_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# 9. 加载模型
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=num_labels
)
model.to(device)

# 10. 优化器
optimizer = AdamW(model.parameters(), lr=5e-5)

# 11. 训练
EPOCHS = 3
for epoch in range(EPOCHS):
    print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
    model.train()
    loop = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())

    # 12. 验证
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, axis=1).cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())

    # 13. 打印准确率
    acc = accuracy_score(true_labels, preds)
    print(f"Epoch {epoch+1} - Accuracy: {acc:.4f}")


# 15. 测试模型
def test_model(model, tokenizer, texts):
    model.eval()
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, axis=1).cpu().numpy()
    return predictions
# 测试模型
test_texts = ["I am so happy today!", "I am very sad."]
predictions = test_model(model, tokenizer, test_texts)
# 打印预测结果
for text, pred in zip(test_texts, predictions):
    print(f"Text: {text} => Predicted label: {pred}")
# 16. 结束
print("Training complete.")

# 14. 保存模型
model.save_pretrained("emotion_model/distilbert-emotion")
tokenizer.save_pretrained("emotion_model/distilbert-emotion")
