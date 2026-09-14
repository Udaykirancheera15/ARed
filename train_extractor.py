#!/usr/bin/env python3
"""
AUTORED - Sensitive Information Extractor (Section IV-B)
Fixed: Memory optimization via AMP & adaptive batch size
"""

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
from config import CFG
from utils import load_jsonl, set_seed, clear_memory

set_seed(42)

# ============================================================
# HYPERPARAMETERS & SETUP
# ============================================================
# Use local cuda:0 inside Python since CUDA_VISIBLE_DEVICES remaps the chosen GPU to 0
# DEVICE = torch.device("cuda:0" if torch.cuda.is me_available() else "cpu")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Reduced per-device batch size to prevent OOM
BATCH_SIZE = getattr(CFG, 'extractor_batch_size', 8)
if BATCH_SIZE > 16:
    BATCH_SIZE = 16  # Safeguard batch size for ~16GB GPUs

GRADIENT_ACCUMULATION_STEPS = 4
MAX_LENGTH = 384
TARGET_MAX_LENGTH = 64
NUM_WORKERS = 2
PIN_MEMORY = True

class ExtractionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=MAX_LENGTH, target_max_length=TARGET_MAX_LENGTH):
        self.samples = []
        for item in data:
            if not item.get('access_code'):
                continue
            
            input_text = f"Extract the exact access code from the following text. If no access code is present, output 'NONE': {item['llm_output']}"
            target_text = item['access_code']
            
            source = tokenizer(
                input_text,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            target = tokenizer(
                target_text,
                max_length=target_max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            labels = target['input_ids'].squeeze(0)
            labels[labels == tokenizer.pad_token_id] = -100
            
            self.samples.append({
                'input_ids': source['input_ids'].squeeze(0),
                'attention_mask': source['attention_mask'].squeeze(0),
                'labels': labels,
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
    }

class ExtractorTrainer:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained(CFG.extractor_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(CFG.extractor_model_name).to(DEVICE)
    
    def train(self, train_data, val_data=None):
        dataset = ExtractionDataset(train_data, self.tokenizer)
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        
        optimizer = AdamW(self.model.parameters(), lr=CFG.extractor_learning_rate)
        scaler = GradScaler()
        accumulation_steps = GRADIENT_ACCUMULATION_STEPS
        self.model.train()
        
        print(f"Total batches per epoch: {len(dataloader)}")
        print(f"Effective batch size: {BATCH_SIZE * accumulation_steps}")
        
        for epoch in range(CFG.extractor_epochs):
            total_loss = 0
            optimizer.zero_grad()
            progress = tqdm(dataloader, desc=f"Extractor Epoch {epoch+1}/{CFG.extractor_epochs}")
            
            for step, batch in enumerate(progress):
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * accumulation_steps
                progress.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
            clear_memory()
        
        os.makedirs(CFG.extractor_path, exist_ok=True)
        self.model.save_pretrained(CFG.extractor_path)
        self.tokenizer.save_pretrained(CFG.extractor_path)
        print(f"Extractor saved to {CFG.extractor_path}")
    
    @torch.no_grad()
    def extract(self, text):
        self.model.eval()
        inputs = self.tokenizer(
            f"Extract the exact access code from the following text. If no access code is present, output 'NONE': {text}",
            max_length=MAX_LENGTH,
            truncation=True,
            return_tensors='pt'
        ).to(DEVICE)
        
        outputs = self.model.generate(
            input_ids=inputs['input_ids'], 
            max_length=TARGET_MAX_LENGTH,
            num_beams=4,
            early_stopping=True
        )
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result if result != "NONE" else None

if __name__ == "__main__":
    data = load_jsonl(CFG.dataset_path)
    
    positives = [item for item in data 
                 if item.get('llm_output') and item.get('access_code')
                 and item['access_code'].lower() in item['llm_output'].lower()]
    
    print(f"Found {len(positives)} positive examples (code literally present)")
    
    if len(positives) == 0:
        print("ERROR: No positive examples found!")
        exit(1)
    
    negatives = [item for item in data
                 if item.get('llm_output') and item.get('access_code')
                 and item['access_code'].lower() not in item['llm_output'].lower()]
    negatives = negatives[:len(positives)]
    
    for item in negatives:
        item['access_code'] = 'NONE'
    
    valid = positives + negatives
    random.shuffle(valid)
    
    print(f"Total training samples: {len(valid)} ({len(positives)} positives, {len(negatives)} negatives)")
    
    split = int(0.9 * len(valid))
    train_data = valid[:split]
    val_data = valid[split:]
    
    trainer = ExtractorTrainer()
    trainer.train(train_data, val_data)
    
    print("\n" + "="*60)
    print("TESTING EXTRACTOR ON POSITIVE EXAMPLES")
    print("="*60)
    
    for i, item in enumerate(positives[:3]):
        extracted = trainer.extract(item['llm_output'])
        print(f"\n--- Test {i+1} ---")
        print(f"True code: {item['access_code']}")
        print(f"Extracted: {extracted}")
        print(f"Success: {extracted == item['access_code']}")