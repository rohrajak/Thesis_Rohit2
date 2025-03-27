import os
import json
import torch
import numpy as np
import gc
from datasets import load_from_disk, concatenate_datasets
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import WeightedRandomSampler
from collections import defaultdict
import psutil
import matplotlib.pyplot as plt

# Optimizations for CPU
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(4)

MODEL_PATH = "models/multitask_model"
REPORT_DIR = "reports/training"
os.makedirs(REPORT_DIR, exist_ok=True)

# Training configuration
TRAINING_CONFIG = {
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "num_train_epochs": 2,
    "gradient_accumulation_steps": 4,
    "dataloader_num_workers": 0,
    "optim": "adafactor",
    "fp16": False,
    "logging_steps": 100,
    "save_steps": 500,
    "eval_steps": 100,
    "eval_accumulation_steps": 2,
    "report_to": "none",
    "disable_tqdm": False,
    "lr_scheduler_type": "constant",
    "learning_rate": 3e-5,
    "weight_decay": 0.0,
    "max_grad_norm": 0.5,
    "load_best_model_at_end": False,     # Disabled for multi-task phase
    "metric_for_best_model": "loss",
    "greater_is_better": False
}

class MultiTaskTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_history = defaultdict(list)
        self.memory_log = []
        
    def compute_loss(self, model, inputs, return_outputs=False):
        gc.collect()
        self.memory_log.append(psutil.virtual_memory().percent)
        
        emotion_labels = inputs.pop("emotion", None)
        if emotion_labels is not None:
            emotion_labels = torch.tensor(
                [model.config.emotion_labels.index(e) for e in emotion_labels],
                dtype=torch.long,
                device=model.device
            )
        
        inputs = self._prepare_inputs(inputs)
        outputs = model(**inputs, emotion_labels=emotion_labels)
        
        if outputs.loss is not None:
            self.loss_history["total_loss"].append(outputs.loss.item())
            if len(self.loss_history["total_loss"]) > 50:
                self.loss_history["total_loss"].pop(0)
                
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def _generate_training_reports(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.loss_history["total_loss"])
        plt.title("Training Loss")
        plt.savefig(os.path.join(REPORT_DIR, "training_loss.png"))
        plt.close()

def load_datasets():
    """Load datasets with limited samples, handling smaller datasets"""
    datasets = {}
    for name in ["daily_dialog", "empathetic_dialogues", "custom_support_tickets"]:
        path = f"data/processed/{name}/train"
        if os.path.exists(path):
            dataset = load_from_disk(path)
            # Get actual size and use minimum of requested size or actual size
            actual_size = len(dataset)
            sample_size = min(2000, actual_size)
            if sample_size < 2000:
                print(f"Warning: Dataset {name} only has {actual_size} samples, using {sample_size} samples")
            datasets[name] = dataset.select(range(sample_size))
    return datasets

def per_dataset_finetuning(model, tokenizer):
    datasets = load_datasets()
    
    for name, train_ds in datasets.items():
        print(f"\n=== Phase 1: {name.replace('_', ' ').title()} ===")
        
        val_path = f"data/processed/{name}/validation"
        if os.path.exists(val_path):
            valid_ds = load_from_disk(val_path)
            # Adjust validation size proportionally
            val_size = min(200, len(valid_ds))
            valid_ds = valid_ds.select(range(val_size))
        else:
            print(f"Warning: No validation set found for {name}")
            continue
        
        args = TrainingArguments(
            output_dir=f"{MODEL_PATH}_{name}",
            **{**TRAINING_CONFIG, "load_best_model_at_end": True},  # Enable for per-dataset
            evaluation_strategy="steps",
            save_strategy="steps"
        )
        
        trainer = MultiTaskTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=valid_ds,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
        )
        
        trainer.train()
        trainer.save_model()
        trainer._generate_training_reports()
        del trainer
        gc.collect()

def multi_task_finetuning(model, tokenizer):
    datasets = load_datasets()
    combined_ds = concatenate_datasets(list(datasets.values()))
    
    args = TrainingArguments(
        output_dir=MODEL_PATH,
        **TRAINING_CONFIG,  # load_best_model_at_end=False by default
        evaluation_strategy="no",
        save_strategy="epoch"  # Changed to match eval strategy
    )
    
    trainer = MultiTaskTrainer(
        model=model,
        args=args,
        train_dataset=combined_ds
    )
    
    try:
        trainer.train()
    except RuntimeError as e:
        print(f"Training error: {e}")
    
    trainer.save_model()
    trainer._generate_training_reports()

def main():
    from src.stepIII_model_architecture import initialize_model
    model, tokenizer = initialize_model()
    
    per_dataset_finetuning(model, tokenizer)
    multi_task_finetuning(model, tokenizer)

if __name__ == "__main__":
    main()