# step3_balancing.py
import json
import numpy as np
from collections import Counter
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk

# Configuration
REPORT_DIR = "reports/balancing"
WEIGHTS_DIR = "data/weights"
DATASET_PATHS = {
    "DailyDialog": "data/processed/daily_dialog/train",
    "EmpatheticDialogues": "data/processed/empathetic_dialogues/train",
    "CustomSupportTickets": "data/processed/custom_support_tickets/train"
}

def main():
    # Create directories
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    
    # Modified: Process datasets sequentially to save memory
    for dataset_name, dataset_path in DATASET_PATHS.items():
        if os.path.exists(dataset_path):
            print(f"Processing {dataset_name}...")
            dataset = load_from_disk(dataset_path)
            balance_dataset(dataset, dataset_name)
            del dataset  # Explicit memory cleanup
        else:
            print(f"Skipping {dataset_name} - processed data not found")

def balance_dataset(dataset, dataset_name):
    """Main balancing function with reporting"""
    # Get class distribution
    emotions = dataset["emotion"]
    class_counts = Counter(emotions)
    total = len(emotions)
    
    # Calculate weights

    class_weights = {
        cls: total/(len(class_counts)*count) / total  # Normalized weights
        for cls, count in class_counts.items()
    }

    #class_weights = {cls: total/(num_classes * count) for cls, count in class_counts.items()}

    # Generate reports and visualizations
    generate_balancing_report(class_counts, class_weights, dataset_name)
    
    # Save weights
    save_weights(class_weights, dataset_name)

def generate_balancing_report(class_counts, class_weights, dataset_name):
    """Generate visual and JSON reports"""
    classes = sorted(class_counts.keys())
    counts = [class_counts[cls] for cls in classes]
    weights = [class_weights[cls] for cls in classes]
    
    # Normalize for visualization
    counts_norm = [c/sum(counts) for c in counts]
    weights_norm = [w/sum(weights) for w in weights]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Before balancing
    plt.subplot(1, 2, 1)
    sns.barplot(x=counts_norm, y=classes, palette="viridis")
    plt.title(f"{dataset_name} Class Distribution\nBefore Balancing")
    plt.xlabel("Normalized Frequency")
    
    # After balancing (weighted)
    plt.subplot(1, 2, 2)
    sns.barplot(x=weights_norm, y=classes, palette="magma")
    plt.title(f"{dataset_name} Class Distribution\nAfter Weighting")
    plt.xlabel("Normalized Weight")
    
    # Save visual report
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f"{dataset_name.lower()}_balancing.png"))
    plt.close()
    
    # Create JSON report
    report = {
        "dataset": dataset_name,
        "total_samples": sum(counts),
        "class_distribution": class_counts,
        "class_weights": class_weights,
        "weighting_strategy": "Normalized inverse frequency weighting"
    }
    
    report_path = os.path.join(REPORT_DIR, f"{dataset_name.lower()}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

def save_weights(class_weights, dataset_name):
    """Save calculated weights to file"""
    weights_path = os.path.join(WEIGHTS_DIR, f"{dataset_name.lower()}_weights.json")
    with open(weights_path, "w") as f:
        json.dump(class_weights, f, indent=2)
    print(f"Saved weights for {dataset_name} to {weights_path}")

if __name__ == "__main__":
    main()