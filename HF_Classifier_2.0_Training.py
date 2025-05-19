import os
import pandas as pd
import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer, BertForSequenceClassification, AdamW,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader, Dataset

# Configuration
EPOCHS = 5
BATCH_SIZE = 16
GRADIENT_ACCUM_STEPS = 2
LEARNING_RATE = 5e-5
MAX_LEN = 512
LABEL_COLUMNS = [
    'Wrong Object', 'Memory failure', 'Fear', 'Distraction', 'Fatigue',
    'Performance Variability', 'Inattention', 'Physiological stress',
    'Psychological stress', 'Functional impairment', 'Cognitive style',
    'Cognitive bias', 'Equipment failure', 'Software fault',
    'Inadequate procedure', 'Access limitations', 'Ambiguous information',
    'Incomplete information', 'Access problems', 'Mislabelling',
    'Communication failure', 'Missing information', 'Maintenance failure',
    'Inadequate quality control', 'Management problem', 'Design failure',
    'Inadequate task allocation', 'Social pressure', 'Insufficient skills',
    'Insufficient knowledge', 'Temperature', 'Sound', 'Humidity',
    'Illumination', 'Other', 'Adverse ambient conditions',
    'Excessive demand', 'Inadequate work place layout',
    'Inadequate team support', 'Irregular working hours'
]
DATA_PATH = 'MATA_D_VirtRaph_Complete.xlsx'
TEXT_COLUMN = 'Accident Description'

# Utility to chunk text into sub-sequences (naive whitespace token split)
def chunk_text(tokenizer, text, max_len):
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_len):
        chunk = tokens[i:i + max_len]
        chunk = tokenizer.convert_tokens_to_string(chunk)
        chunks.append(chunk)
    return chunks

# Updated chunking function with report_id tracking
def chunk_reports(df, label_col, tokenizer, max_len):
    input_ids, attention_masks, labels, report_ids = [], [], [], []

    for report_id, row in enumerate(df.itertuples()):
        chunks = chunk_text(tokenizer, getattr(row, TEXT_COLUMN), max_len - 2)
        encoding = tokenizer(chunks, truncation=True, padding=True, return_tensors='pt', max_length=max_len)

        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])
        labels.extend([getattr(row, label_col)] * len(chunks))
        report_ids.extend([report_id] * len(chunks))

    return (
        torch.cat(input_ids),
        torch.cat(attention_masks),
        torch.tensor(labels),
        torch.tensor(report_ids)
    )

# Custom dataset that returns report_id per chunk
class ChunkedReportDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels, report_ids):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.report_ids = report_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.input_ids[idx],
            self.attention_masks[idx],
            self.labels[idx],
            self.report_ids[idx]
        )

# Setup
df = pd.read_excel(DATA_PATH)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train a model per label
for label_col in LABEL_COLUMNS:
    print(f"\nTraining model for: {label_col}")
    train_df, val_df = train_test_split(df[[TEXT_COLUMN, label_col]], test_size=0.1)

    train_inputs, train_masks, train_labels, train_rids = chunk_reports(train_df, label_col, tokenizer, MAX_LEN)
    val_inputs, val_masks, val_labels, val_rids = chunk_reports(val_df, label_col, tokenizer, MAX_LEN)

    train_dataset = ChunkedReportDataset(train_inputs, train_masks, train_labels, train_rids)
    val_dataset = ChunkedReportDataset(val_inputs, val_masks, val_labels, val_rids)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * EPOCHS
    )
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"{label_col} - Epoch {epoch+1}/{EPOCHS}")):
            input_ids, attention_mask, labels, report_ids = [b.to(device) for b in batch]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # (batch_size, 2)

            # Group logits by report_id
            report_logits = defaultdict(list)
            report_targets = {}
            for i, rid in enumerate(report_ids):
                report_logits[rid.item()].append(logits[i])
                report_targets[rid.item()] = labels[i]

            # Average logits per report and calculate loss
            report_losses = []
            for rid, logit_list in report_logits.items():
                avg_logit = torch.stack(logit_list).mean(dim=0, keepdim=True)
                label = report_targets[rid].unsqueeze(0)
                loss = F.cross_entropy(avg_logit, label)
                report_losses.append(loss)

            loss = torch.stack(report_losses).mean() / GRADIENT_ACCUM_STEPS
            scaler.scale(loss).backward()

            if (step + 1) % GRADIENT_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            total_loss += loss.item() * GRADIENT_ACCUM_STEPS

        avg_loss = total_loss / len(train_loader)
        print(f"{label_col} - Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    save_path = f"Models/BERT_{label_col.replace(' ', '')}"
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved model for {label_col} to {save_path}")
