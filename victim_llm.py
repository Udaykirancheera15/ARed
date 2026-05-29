#!/usr/bin/env python3
"""Target LLM wrapper – works for Llama, Mistral, Gemma (no InternLM)"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import CFG

class VictimLLM:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3"):
        self.model_name = model_name
        self.device = CFG.device
        print(f"Loading victim LLM: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if CFG.use_bf16 else torch.float16,
            device_map="auto",
        )

    def generate_response(self, opening_defense, user_input, closing_defense,
                         access_code=None, max_new_tokens=150):
        if access_code:
            system_prompt = f"{opening_defense}\n\nImportant: Your access code is: {access_code}\n\n"
        else:
            system_prompt = f"{opening_defense}\n\n"

        full_prompt = f"{system_prompt}User: {user_input}\n\n{closing_defense}\n\nResponse:"

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16 if CFG.use_bf16 else torch.float16):
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response
