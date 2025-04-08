import torch
import pandas as pd
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


dataset = load_from_disk("/Users/echohe/Desktop/FeelTrip/emotion_model/combined_emotion_dataset")

dataset = dataset["train"] if "train" in dataset else dataset
df = dataset.to_pandas()
print("Column names:", df.columns)
print(df.head())


texts = df["text"].tolist()
labels = df["label"].tolist()

num_labels = len(set(labels))

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Define the dataset class
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

# Create the dataset and dataloader
train_dataset = EmotionDataset(train_texts, train_labels)
val_dataset = EmotionDataset(val_texts, val_labels)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)


model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=num_labels
)
model.to(device)


optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
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


    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, axis=1).cpu().numpy())
            true_labels.extend(batch["labels"].cpu().numpy())

# Calculate accuracy
    acc = accuracy_score(true_labels, preds)
    print(f"Epoch {epoch+1} - Accuracy: {acc:.4f}")


#test the model
def test_model(model, tokenizer, texts):
    model.eval()
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, axis=1).cpu().numpy()
    return predictions

test_texts = ["I am so happy today!", "I am very sad."]
predictions = test_model(model, tokenizer, test_texts)
# Print the predictions
for text, pred in zip(test_texts, predictions):
    print(f"Text: {text} => Predicted label: {pred}")

print("Training complete.")

#Save the model and tokenizer
model.save_pretrained("emotion_model/distilbert-emotion")
tokenizer.save_pretrained("emotion_model/distilbert-emotion")
