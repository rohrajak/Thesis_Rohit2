import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from rouge_score import rouge_scorer

class ModelEvaluator:
    def __init__(self, model_path, tokenizer_path, data_paths):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.data_paths = {k:v for k,v in data_paths.items() if os.path.exists(v)}
        self.metrics_store = defaultdict(dict)
        self.experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(f"reports/evaluation/{self.experiment_id}", exist_ok=True)

    def compute_metrics(self):
        """Main evaluation method with proper tensor shape handling"""
        for dataset_name, data_path in self.data_paths.items():
            try:
                dataset = load_from_disk(data_path)
                if not all(col in dataset.column_names for col in ['input_ids', 'attention_mask', 'labels', 'emotion']):
                    print(f"Skipping {dataset_name} - missing required columns")
                    continue

                # Process first 100 samples only for evaluation
                eval_dataset = dataset.select(range(min(100, len(dataset))))
                self._compute_language_metrics(eval_dataset, dataset_name)
                self._compute_emotion_metrics(eval_dataset, dataset_name)
                self._compute_perplexity(eval_dataset, dataset_name)
            except Exception as e:
                print(f"Error processing {dataset_name}: {str(e)}")
        
        self._save_metrics()
        self._generate_visualizations()

    def _compute_language_metrics(self, dataset, dataset_name):
        """Calculate ROUGE scores with proper batch handling"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        predictions, references = [], []
        
        batch_size = 4  # Fixed batch size
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            current_batch_size = len(batch["input_ids"])
            
            # Prepare inputs
            inputs = {
                "input_ids": torch.tensor(batch["input_ids"]).to(self.device),
                "attention_mask": torch.tensor(batch["attention_mask"]).to(self.device)
            }
            
            # Generate responses
            outputs = self.model.generate(
                **inputs,
                max_length=64,
                num_beams=1,
                do_sample=False
            )
            preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Prepare references
            refs = []
            for label_ids in batch["labels"]:
                label_ids = np.array(label_ids)
                label_ids = label_ids[label_ids != -100]
                refs.append(self.tokenizer.decode(label_ids, skip_special_tokens=True))
            
            predictions.extend(preds)
            references.extend(refs)
            
            # Calculate ROUGE
            for ref, pred in zip(refs, preds):
                scores = scorer.score(ref, pred)
                for key in scores:
                    self.metrics_store[dataset_name].setdefault(key, []).append(scores[key].fmeasure)

        # Average ROUGE scores
        for key in ['rouge1', 'rougeL']:
            if key in self.metrics_store[dataset_name]:
                self.metrics_store[dataset_name][key] = np.mean(self.metrics_store[dataset_name][key])

    def _compute_emotion_metrics(self, dataset, dataset_name):
        """Calculate emotion classification metrics"""
        true_labels, pred_labels = [], []
        
        batch_size = 4  # Fixed batch size
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            current_batch_size = len(batch["input_ids"])
            
            # Convert to tensors
            inputs = {
                "input_ids": torch.tensor(batch["input_ids"]).to(self.device),
                "attention_mask": torch.tensor(batch["attention_mask"]).to(self.device)
            }
            
            with torch.no_grad():
                # Get encoder outputs only
                encoder_outputs = self.model.get_encoder()(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"]
                )
                
                # Emotion prediction
                pooled = encoder_outputs.last_hidden_state.mean(dim=1)
                preds = torch.argmax(pooled, dim=1).cpu().numpy()
                
                # Get true emotion labels
                try:
                    if isinstance(batch["emotion"][0], str):
                        true_emotions = [self.model.config.emotion_labels.index(e) 
                                       for e in batch["emotion"]]
                    else:
                        true_emotions = batch["emotion"]
                except (AttributeError, ValueError):
                    continue
                
                pred_labels.extend(preds)
                true_labels.extend(true_emotions)
        
        if true_labels and pred_labels:
            report = classification_report(true_labels, pred_labels, output_dict=True)
            self.metrics_store[dataset_name].update({
                "accuracy": report["accuracy"],
                "f1_weighted": report["weighted avg"]["f1-score"],
                "confusion_matrix": confusion_matrix(true_labels, pred_labels)
            })

    def _compute_perplexity(self, dataset, dataset_name):
        """Calculate perplexity with proper decoder inputs"""
        losses = []
        batch_size = 4
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            current_batch_size = len(batch["input_ids"])
            
            # Prepare input tensors
            input_ids = torch.tensor(batch["input_ids"]).to(self.device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(self.device)
            labels = torch.tensor(batch["labels"]).to(self.device)
            
            # Replace -100 with pad_token_id and generate decoder inputs
            labels[labels == -100] = self.tokenizer.pad_token_id
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=labels)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels
                )
                losses.append(outputs.loss.item())
        
        if losses:
            self.metrics_store[dataset_name]["perplexity"] = np.exp(np.mean(losses))

    def _save_metrics(self):
        """Save metrics to files"""
        with open(f"reports/evaluation/{self.experiment_id}/metrics.json", "w") as f:
            json.dump(self.metrics_store, f, indent=2)
            
        pd.DataFrame([
            {"dataset": k, **v} for k,v in self.metrics_store.items()
        ]).to_csv(f"reports/evaluation/{self.experiment_id}/comparison.csv", index=False)

    def _generate_visualizations(self):
        """Generate confusion matrix plots"""
        for dataset_name, metrics in self.metrics_store.items():
            if "confusion_matrix" in metrics:
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    metrics["confusion_matrix"],
                    annot=True,
                    fmt="d",
                    xticklabels=getattr(self.model.config, "emotion_labels", range(len(metrics["confusion_matrix"]))),
                    yticklabels=getattr(self.model.config, "emotion_labels", range(len(metrics["confusion_matrix"])))
                )
                plt.title(f"Confusion Matrix - {dataset_name}")
                plt.savefig(f"reports/evaluation/{self.experiment_id}/confusion_matrix_{dataset_name}.png")
                plt.close()

if __name__ == "__main__":
    # Configuration - update these paths as needed
    CONFIG = {
        "MODEL_PATH": "models/multitask_model",
        "TOKENIZER_PATH": "data/processed/tokenizer",
        "DATA_PATHS": {
            "daily_dialog": "data/processed/daily_dialog/test",
            "empathetic": "data/processed/empathetic_dialogues/test",
            "custom": "data/processed/custom_support_tickets/test"
        }
    }
    
    evaluator = ModelEvaluator(
        CONFIG["MODEL_PATH"],
        CONFIG["TOKENIZER_PATH"],
        CONFIG["DATA_PATHS"]
    )
    evaluator.compute_metrics()