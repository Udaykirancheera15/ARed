#!/usr/bin/env python3
"""
AUTORED - Supervised Fine-Tuning (Section IV-A)
Uses the 12k Mistral-generated dataset (attacker_input as target)
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from config import CFG
from utils import load_jsonl, set_seed, clear_memory

set_seed(42)

class SFTDataset(Dataset):
    """Dataset from the 12k JSONL file"""
    def __init__(self, data, tokenizer, max_length=256, target_max_length=128):
        self.samples = []
        for item in data:
            context = f"Generate an attack prompt to extract the access code. Opening defense: {item['opening_defense']} Closing defense: {item['closing_defense']}"
            target = item['attacker_input']
            
            source_enc = tokenizer(
                context, 
                max_length=max_length, 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt'
            )
            target_enc = tokenizer(
                target, 
                max_length=target_max_length, 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt'
            )
            
            labels = target_enc['input_ids'].squeeze(0)
            labels[labels == tokenizer.pad_token_id] = -100
            
            self.samples.append({
                'input_ids': source_enc['input_ids'].squeeze(0),
                'attention_mask': source_enc['attention_mask'].squeeze(0),
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

class SFTTrainer:
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained(CFG.sft_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(CFG.sft_model_name).to(CFG.device)
    
    def train(self, train_data, val_data=None):
        dataset = SFTDataset(train_data, self.tokenizer, max_length=CFG.sft_max_length)
        dataloader = DataLoader(
                        dataset,
                        batch_size=CFG.sft_batch_size,
                        shuffle=True,
                        collate_fn=collate_fn,
                        num_workers=4,          # parallel data loading
                        pin_memory=True         # faster transfer to GPU
                    )
                            
        optimizer = AdamW(self.model.parameters(), lr=CFG.sft_learning_rate)
        total_steps = len(dataloader) * CFG.sft_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
        
        self.model.train()
        for epoch in range(CFG.sft_epochs):
            total_loss = 0
            progress = tqdm(dataloader, desc=f"SFT Epoch {epoch+1}/{CFG.sft_epochs}")
            for batch in progress:
                input_ids = batch['input_ids'].to(CFG.device)
                attention_mask = batch['attention_mask'].to(CFG.device)
                labels = batch['labels'].to(CFG.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                progress.set_postfix({'loss': f"{loss.item():.4f}"})
                clear_memory()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
        
        os.makedirs(CFG.sft_model_path, exist_ok=True)
        self.model.save_pretrained(CFG.sft_model_path)
        self.tokenizer.save_pretrained(CFG.sft_model_path)
        print(f"SFT model saved to {CFG.sft_model_path}")
    
    @torch.no_grad()
    def generate_attack(self, opening_defense, closing_defense, max_length=100):
        self.model.eval()
        context = f"Generate an attack prompt to extract the access code. Opening defense: {opening_defense} Closing defense: {closing_defense}"
        inputs = self.tokenizer(context, max_length=CFG.sft_max_length, truncation=True, return_tensors='pt').to(CFG.device)
        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("Loading dataset...")
    data = load_jsonl(CFG.dataset_path)
    print(f"Loaded {len(data)} samples")
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    trainer = SFTTrainer()
    trainer.train(train_data, val_data)
    test_item = data[0]
    attack = trainer.generate_attack(test_item['opening_defense'], test_item['closing_defense'])
    print(f"\nGenerated attack:\n{attack}")
