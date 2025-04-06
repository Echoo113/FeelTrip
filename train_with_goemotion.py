from datasets import load_dataset, concatenate_datasets, Features, Value
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import numpy as np
from tqdm import tqdm


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🧠 Using device: {device}")

# === 1. 读取数据集 ===
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
reverse_map = {label: group for group, labels in emotion_map.items() for label in labels}

def map_labels(example):
    new_labels = [reverse_map[label_names[i]] for i in example["labels"] if label_names[i] in reverse_map]
    example["label"] = new_labels[0] if new_labels else None
    return example

goemotions = goemotions.map(map_labels)
goemotions = goemotions.filter(lambda x: x["label"] is not None)

label_to_id = {label: i for i, label in enumerate(["sadness", "joy", "love", "anger", "fear", "surprise"])}
goemotions = goemotions.map(lambda x: {"label": label_to_id[x["label"]]})
goemotions = goemotions.remove_columns(set(goemotions.column_names) - {"text", "label"})

# 修正 cast() 参数类型
emotion_ds = emotion_ds.cast(Features({
    "text": Value("string"),
    "label": Value("int64")
}))

# === 2. 合并并划分数据 ===
combined = concatenate_datasets([emotion_ds, goemotions]).shuffle(seed=42)
split = combined.train_test_split(test_size=0.1)
train_ds = split["train"].select(range(2000))  # 选前2000条用于快速训练
test_ds = split["test"].select(range(500))     # 选前500条作为测试

# === 3. 分词器与编码 ===
model_name = "distilbert-base-uncased"  # 更轻更快
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

columns = ["input_ids", "attention_mask", "label"]
train_ds.set_format("torch", columns=columns)
test_ds.set_format("torch", columns=columns)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

# === 4. 加载模型 ===
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# === 5. 自定义训练 + 测试 ===
def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()
for epoch in range(5):
    print(f"\n🔥 Epoch {epoch+1}")
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"]  # ✅ 这里必须是 labels！
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"✅ Training Loss: {avg_loss:.4f}")
#evaluate
# === Evaluation ===
model.eval()
accs = []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        preds = torch.argmax(outputs.logits, dim=1)
        acc = (preds == batch["label"]).float().mean().item()
        accs.append(acc)

print(f"🎯 Test Accuracy: {np.mean(accs):.4f}")
