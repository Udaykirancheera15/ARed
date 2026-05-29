#!/usr/bin/env python3
"""
AUTORED - Stop Point Identifier (Section IV-C)
Binary classifier: does the response contain the access code?
Uses real llm_output and output_is_access_granted from dataset.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from config import CFG
from utils import load_jsonl, set_seed, clear_memory

set_seed(42)

class DetectorDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.samples = []
        for item in data:
            text = item['llm_output']
            # Use the dataset's ground truth: output_is_access_granted
            label = 1 if item.get('output_is_access_granted', False) else 0
            enc = tokenizer(text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
            self.samples.append({
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long),
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch]),
    }

class DetectorTrainer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(CFG.detector_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(CFG.detector_model_name, num_labels=2).to(CFG.device)
    
    def train(self, train_data, val_data=None):
        dataset = DetectorDataset(train_data, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=CFG.detector_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=16,pin_memory=True,prefetch_factor=4)
        optimizer = AdamW(self.model.parameters(), lr=CFG.detector_learning_rate)
        
        self.model.train()
        for epoch in range(CFG.detector_epochs):
            total_loss = 0
            correct = 0
            total = 0
            progress = tqdm(dataloader, desc=f"Detector Epoch {epoch+1}")
            for batch in progress:
                input_ids = batch['input_ids'].to(CFG.device)
                attention_mask = batch['attention_mask'].to(CFG.device)
                labels = batch['label'].to(CFG.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                progress.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{correct/total:.3f}"})
                clear_memory()
            
            avg_loss = total_loss / len(dataloader)
            acc = correct / total
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}")
        
        os.makedirs(CFG.detector_path, exist_ok=True)
        self.model.save_pretrained(CFG.detector_path)
        self.tokenizer.save_pretrained(CFG.detector_path)
        print(f"Detector saved to {CFG.detector_path}")
    
    @torch.no_grad()
    def should_stop(self, text):
        self.model.eval()
        inputs = self.tokenizer(text, max_length=256, truncation=True, return_tensors='pt').to(CFG.device)
        outputs = self.model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        return pred == 1

if __name__ == "__main__":
    data = load_jsonl(CFG.dataset_path)
    # Use samples that have llm_output
    valid = [item for item in data if item.get('llm_output')]
    print(f"Loaded {len(valid)} detector examples")
    split = int(0.9 * len(valid))
    train_data = valid[:split]
    val_data = valid[split:]
    
    trainer = DetectorTrainer()
    trainer.train(train_data, val_data)
    
    # Test
    test_text = valid[0]['llm_output']
    print(f"Should stop: {trainer.should_stop(test_text)} (true label: {valid[0]['output_is_access_granted']})")
