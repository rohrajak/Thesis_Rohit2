# load_preprocessing.py

import os
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch  # Added for memory optimizations

# Modified: Increased optimization settings for CPU
os.environ["OMP_NUM_THREADS"] = "4"  # Increased from 2
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(4)

# Configuration
MODEL_NAME = "google/t5-efficient-tiny" # Keep tokenizer consistent
SPECIAL_TOKENS = [
    "[General]", "[Empathy]", "[Support]",
    "[Emotion:", "]", "[Dialog]:", "[Context]:", "[Ticket]:"
]
# Modified: Reduced emotion categories
EMOTION_MAP = {
    0: "neutral", 1: "anger", 2: "happy", 
    3: "sad", 4: "fear"  # Consolidated emotions
}

# Initialize tokenizer with special tokens
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
TOKENIZER.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

def preprocess_data():
    create_directories()
    process_daily_dialog()
    process_empathetic_dialogues()
    process_custom_support_tickets()
    TOKENIZER.save_pretrained("data/processed/tokenizer")
    print("Tokenizer saved with special tokens")

def create_directories():
    os.makedirs("data/processed/daily_dialog", exist_ok=True)
    os.makedirs("data/processed/empathetic_dialogues", exist_ok=True)
    os.makedirs("data/processed/custom_support_tickets", exist_ok=True)
    os.makedirs("visualizations/emotion_distribution", exist_ok=True)
    os.makedirs("data/cache", exist_ok=True)  

# --------------------------
# DailyDialog Processing
# --------------------------

def process_daily_dialog():
    # Modified: Load dataset with streaming to save memory
    dataset = load_dataset("daily_dialog", streaming=True)
    processed = DatasetDict()
    
    for split in ["train", "validation", "test"]:
        if split not in dataset: continue
        
        # Modified: Process in smaller chunks
        df = pd.DataFrame(list(dataset[split].take(2000)))  # Limited samples
        df = preprocess_daily_dialog_df(df)
        dataset_split = process_split(Dataset.from_pandas(df), "daily_dialog")
        processed[split] = dataset_split
        save_dataset(dataset_split, f"daily_dialog/{split}")
    
    generate_visualizations(processed["train"], "DailyDialog")

def preprocess_daily_dialog_df(df):
    """Process DailyDialog into input-response pairs"""
    processed_rows = []
    for _, row in df.iterrows():
        dialog = row["dialog"]
        emotions = row["emotion"]
        # Create consecutive utterance pairs
        for i in range(len(dialog)-1):
            # Modified: Simplified emotion mapping
            emotion = EMOTION_MAP.get(emotions[i], "neutral")
            processed_rows.append({
                "input": dialog[i],
                "response": dialog[i+1],
                "emotion": emotion
            })
    return pd.DataFrame(processed_rows).assign(
        text=lambda x: x.apply(
            lambda r: f"[General] [Emotion: {r['emotion']}] [Dialog]: {r['input']}",
            axis=1
        )
    )[["text", "response", "emotion"]]

# --------------------------
# Empathetic Dialogues Processing
# --------------------------

def process_empathetic_dialogues():
    # Modified: Load dataset with streaming
    dataset = load_dataset("empathetic_dialogues", streaming=True)
    processed = DatasetDict()
    
    for split in ["train", "validation", "test"]:
        if split not in dataset: continue
        
        # Modified: Process in smaller chunks
        df = pd.DataFrame(list(dataset[split].take(2000)))  # Limited samples
        df = preprocess_empathetic_df(df)
        dataset_split = process_split(Dataset.from_pandas(df), "empathetic")
        processed[split] = dataset_split
        save_dataset(dataset_split, f"empathetic_dialogues/{split}")
    
    generate_visualizations(processed["train"], "EmpatheticDialogues")

def preprocess_empathetic_df(df):
    """Process EmpatheticDialogues with correct column structure"""
    return df.rename(columns={
        "context": "emotion",
        "prompt": "input",
        "utterance": "response"
    }).assign(
        emotion=lambda x: x["emotion"].str.lower(),
        text=lambda x: x.apply(
            lambda r: f"[Empathy] [Emotion: {r['emotion']}] [Context]: {r['input']}",
            axis=1
        )
    )[["text", "response", "emotion"]]

# --------------------------
# Custom Support Tickets Processing
# --------------------------

def process_custom_support_tickets():
    # Modified: Load only first 2000 rows
    df = pd.read_csv("data/raw/ISP_Customer_Support_Tickets_Dialogs.csv", 
                    nrows=2000, keep_default_na=False, quotechar='"')

    # Process using the dedicated preprocessor
    df_processed = preprocess_custom_df(df)
    
    # Split data
    train, test = train_test_split(df_processed, test_size=0.2, random_state=42)
    train, val = train_test_split(train, test_size=0.1, random_state=42)
    
    # Process and save splits
    for name, split in zip(["train", "validation", "test"], [train, val, test]):
        dataset = process_split(Dataset.from_pandas(split), "custom")
        save_dataset(dataset, f"custom_support_tickets/{name}")
    
    generate_visualizations(Dataset.from_pandas(train), "CustomSupportTickets")

def preprocess_custom_df(df):
    """Process Custom Support Tickets with proper text column creation"""
    processed = []
    for _, row in df.iterrows():
        parts = row["text"].split("\nAgent: ")
        if len(parts) > 1:
            input_text = parts[0].replace("Customer: ", "").strip()
            response_text = parts[1].strip()
            emotion = row["emotion"].lower()
            
            processed.append({
                "text": f"[Support] [Emotion: {emotion}] [Ticket]: {input_text}",
                "response": response_text,
                "emotion": emotion
            })
    
    return pd.DataFrame(processed)

# --------------------------
# Common Processing Functions
# --------------------------

def process_split(dataset, dataset_type):
    def tokenize_fn(examples):

        # Tokenize inputs
        inputs = TOKENIZER(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=64,  # Reduced from 128
            return_tensors="np"
        )
        
        # Tokenize responses using text_target parameter
        labels = TOKENIZER(
            text_target=examples["response"],
            truncation=True,
            padding="max_length",
            max_length=64,  # Reduced from 128
            return_tensors="np"
        )["input_ids"]
        labels = np.where(labels != TOKENIZER.pad_token_id, labels, -100)
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
            "emotion": examples["emotion"]
        }
    
    # Modified: Smaller batch size and disabled caching
    return dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=4,  # Reduced from 8
        remove_columns=dataset.column_names
    )

def save_dataset(dataset, save_path):
    full_path = os.path.join("data/processed", save_path)
    os.makedirs(full_path, exist_ok=True)
    dataset.save_to_disk(full_path)

def sample_check(dataset, split_name, dataset_type):
    """Validate special token preservation"""
    print(f"\n{dataset_type} {split_name} Sample Check:")
    for i in range(2):
        sample = dataset[i]
        print(f"Original: {sample['text']}")
        print(f"Tokenized: {TOKENIZER.decode(sample['input_ids'])}")
        print("Special Tokens Present:", [tok for tok in SPECIAL_TOKENS if tok in sample['text']])
        print("-" * 50)

def generate_visualizations(dataset, dataset_name):
    df = dataset.to_pandas()
    plt.figure(figsize=(10, 6))
    sns.countplot(x="emotion", data=df, order=df["emotion"].value_counts().index)
    plt.title(f"Emotion Distribution in {dataset_name}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"visualizations/emotion_distribution/{dataset_name.lower().replace(' ', '_')}.png")
    plt.close()

if __name__ == "__main__":
    preprocess_data()