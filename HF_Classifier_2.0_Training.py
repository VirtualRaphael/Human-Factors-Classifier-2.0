
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Configuration
EPOCHS = 5
BATCH_SIZE = 16
GRADIENT_ACCUM_STEPS = 2
LEARNING_RATE = 5e-5
MAX_LEN = 512
LABEL_COLUMNS = ['Insufficient skills', 'Wrong reasoning', 'Time pressure']  # Replace with actual column names
DATA_PATH = 'MATA_D_VirtRaph_Complete.xlsx'
TEXT_COLUMN = 'Accident Description'

# Load dataset
df = pd.read_excel(DATA_PATH)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for label_col in LABEL_COLUMNS:
    print(f"Training model for: {label_col}")

    # Split data
    train_df, val_df = train_test_split(df[[TEXT_COLUMN, label_col]], test_size=0.1)

    # Tokenize text
    train_encodings = tokenizer(train_df[TEXT_COLUMN].tolist(), truncation=True, padding=True, return_tensors='pt', max_length=MAX_LEN)
    val_encodings = tokenizer(val_df[TEXT_COLUMN].tolist(), truncation=True, padding=True, return_tensors='pt', max_length=MAX_LEN)

    # Prepare tensors
    train_labels = torch.tensor(train_df[label_col].values)
    val_labels = torch.tensor(val_df[label_col].values)

    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Load model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"{label_col} - Epoch {epoch + 1}/{EPOCHS}")):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / GRADIENT_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRADIENT_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_loss += loss.item() * GRADIENT_ACCUM_STEPS

        avg_loss = total_loss / len(train_loader)
        print(f"{label_col} - Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    # Save model
    save_path = f"Models/BERT_{label_col.replace(' ', '')}"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved model for {label_col} to {save_path}")
