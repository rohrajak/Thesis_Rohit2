from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer
import torch
import torch.nn as nn
from transformers.modeling_outputs import Seq2SeqLMOutput
import json
import os

class MultiTaskEmotionModel(T5ForConditionalGeneration):
    def __init__(self, config, tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.class_weights = None
        self.emotion_labels = self.load_combined_emotions()

        # Emotion detection head
        self.emotion_head = nn.Sequential(
            nn.Linear(config.d_model, len(self.emotion_labels))
        )
        
        self.load_class_weights()
        self.init_special_token_embeddings()
        
        # Task head
        self.task_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Freeze first 4 encoder layers
        for layer in self.encoder.block[:4]:
            for param in layer.parameters():
                param.requires_grad = False

    def init_special_token_embeddings(self):
        """Initialize special token embeddings"""
        for token in ['[General]', '[Empathy]', '[Support]']:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id != self.tokenizer.unk_token_id:
                with torch.no_grad():
                    self.shared.weight[token_id] = self.shared.weight.mean(dim=0)

    def load_combined_emotions(self):
        """Load combined emotions from reports"""
        emotion_sources = [
            "reports/balancing/dailydialog_report.json",
            "reports/balancing/empatheticdialogues_report.json",
            "reports/balancing/customsupporttickets_report.json"
        ]
        
        combined = set()
        for path in emotion_sources:
            try:
                with open(path) as f:
                    report = json.load(f)
                    combined.update(report["class_distribution"].keys())
            except Exception as e:
                print(f"Warning: Could not load emotion report from {path}. Error: {str(e)}")
                continue
        
        if not combined:
            return ["neutral", "anger", "happy", "sad", "fear"]
        return sorted(combined)

    def load_class_weights(self):
        """Load class weights from files"""
        default_weights = {e: 1.0 for e in self.emotion_labels}
        weight_files = [
            "data/weights/dailydialog_weights.json",
            "data/weights/empatheticdialogues_weights.json",
            "data/weights/customsupporttickets_weights.json"
        ]
        
        for path in weight_files:
            try:
                with open(path) as f:
                    weights = json.load(f)
                    default_weights.update({
                        k: float(v) for k,v in weights.items() 
                        if k in self.emotion_labels
                    })
            except Exception as e:
                print(f"Warning: Could not load weights from {path}. Error: {str(e)}")
                continue
        
        weight_tensor = [default_weights[e] for e in self.emotion_labels]
        self.class_weights = torch.tensor(weight_tensor, dtype=torch.float32)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Shared encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Emotion detection
        pooled_encoder = encoder_outputs.last_hidden_state.mean(dim=1)
        emotion_logits = self.emotion_head(pooled_encoder)
        
        # Base T5 processing
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            encoder_outputs=encoder_outputs,
            return_dict=True
        )
        
        # Apply task head
        task_logits = self.task_head(encoder_outputs.last_hidden_state)
        outputs.logits = (task_logits + outputs.logits) * 0.5
        
        # Emotion loss (only if emotion_labels are provided)
        if "emotion_labels" in kwargs and kwargs["emotion_labels"] is not None:
            emotion_labels = kwargs["emotion_labels"]
            if self.class_weights is not None:
                emotion_loss = nn.CrossEntropyLoss(weight=self.class_weights.to(emotion_logits.device))(
                    emotion_logits, emotion_labels
                )
                outputs.loss += 0.3 * emotion_loss
                
        return outputs

    def detect_task_token(self, input_ids):
        """Detect task token from input"""
        token_ids = [
            self.tokenizer.convert_tokens_to_ids(t) 
            for t in ['[General]', '[Empathy]', '[Support]']
        ]
        
        seq = input_ids[0].tolist()
        for tok_id in token_ids:
            if tok_id in seq:
                return self.tokenizer.convert_ids_to_tokens(tok_id)
        return '[General]'

def initialize_model():
    """Initialize model with proper configuration"""
    tokenizer = AutoTokenizer.from_pretrained("data/processed/tokenizer")

    config = T5Config.from_pretrained("google/t5-efficient-tiny",
        d_model=128,
        d_ff=256,
        num_layers=4,
        num_heads=2,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer)
    )
    
    model = MultiTaskEmotionModel(config, tokenizer)
    print("Model initialized with reduced architecture")
    return model, tokenizer