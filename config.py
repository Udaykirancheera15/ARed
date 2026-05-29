#!/usr/bin/env python3
"""Global configuration for AUTORED framework"""

import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class AUTOREDConfig:
    """Global AUTORED configuration"""
    
    # ============================================================
    # PATHS
    # ============================================================
    sft_model_path: str = "outputs/sft/best_model"
    rl_model_path: str = "outputs/rl/best_model"
    extractor_path: str = "outputs/extractor/best_model"
    detector_path: str = "outputs/detector/best_model"
    dataset_path: str = "autored_dataset_base.jsonl"
    
    # ============================================================
    # SFT (Malicious Prompt Generator - Stage 1)
    # ============================================================
    sft_model_name: str = "google/flan-t5-base"
    sft_learning_rate: float = 3e-5
    sft_epochs: int = 10
    sft_batch_size: int = 16
    sft_max_length: int = 256
    
    # ============================================================
    # RL MODULE (NLPO/PPO)
    # ============================================================
    rl_model_name: str = "google/flan-t5-base"
    rl_learning_rate: float = 1e-7                    # Reduced for stability
    rl_epochs: int = 10
    rl_batch_size: int = 16
    rl_max_steps: int = 30
    rl_ppo_epochs: int = 4
    rl_clip_epsilon: float = 0.2
    rl_gamma: float = 0.99
    rl_gae_lambda: float = 0.95
    rl_entropy_coef: float = 0.01
    rl_value_coef: float = 0.5
    rl_top_p_mask: float = 0.9
    rl_reward_scale: float = 1.0
    rl_num_rollouts_per_update: int = 4              # Reduced for stability
    
    # ============================================================
    # RL OPTIMIZED SETTINGS
    # ============================================================
    rl_num_workers: int = 0                          # Disabled for stability
    rl_use_amp: bool = False                         # Disabled for stability
    rl_gradient_checkpointing: bool = False          # Disabled
    rl_victim_batch_size: int = 1
    
    # ============================================================
    # SENSITIVE INFO EXTRACTOR
    # ============================================================
    extractor_model_name: str = "google/flan-t5-base"
    extractor_learning_rate: float = 2e-5
    extractor_epochs: int = 10
    extractor_batch_size: int = 48
    
    # ============================================================
    # STOP POINT IDENTIFIER
    # ============================================================
    detector_model_name: str = "bert-base-uncased"
    detector_learning_rate: float = 2e-5
    detector_epochs: int = 10
    detector_batch_size: int = 48
    
    # ============================================================
    # CTF GAME
    # ============================================================
    max_interactions: int = 30
    ctf_rounds: int = 70
    access_code_patterns: List[str] = field(default_factory=lambda: [
        "access code", "password", "secret key", "api key",
        "token", "flag", "credential", "private key"
    ])
    
    # ============================================================
    # DEVICE & PRECISION
    # ============================================================
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16: bool = False                           # Disabled for consistency
    use_gradient_checkpointing: bool = False

CFG = AUTOREDConfig()
