"""Simple PyTorch training framework for diagnostic prediction."""

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from diagnostic_prediction_dataloader import DiagnosticPredictionDataset
from diagnostic_data_prep import prepare_diagnostic_data
from pathlib import Path


def load_data(json_path):
    """Load diagnostic data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def collate_fn(batch):
    """Custom collate function to handle batch processing."""
    # Extract relevant fields
    posts = [item['x_post'] for item in batch]
    symptoms_raw = [item.get('x_symptoms_raw', '') for item in batch]
    symptoms_normalized = [item.get('x_symptoms_normalized', []) for item in batch]
    labels = [item['y_diag'][0][0] if item['y_diag'] else 'unknown' for item in batch]
    
    return {
        'posts': posts,
        'symptoms_raw': symptoms_raw,
        'symptoms_normalized': symptoms_normalized,
        'labels': labels,
        'raw': batch  # Keep raw data for inspection
    }


def main():
    # Configuration
    RAW_DATA_PATH = '/Users/robertdoherty/Desktop/Playground/research agent/output/2025-11-03/diagnostic_dataset_2025-11-03_20-59-20.json'
    NORMALIZED_DATA_PATH = '/Users/robertdoherty/Desktop/Playground/research agent/output/2025-11-03/diagnostic_dataset_normalized.json'
    BATCH_SIZE = 8
    TRAIN_SPLIT = 0.8
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Step 1: Prepare normalized data (if needed)
    if not Path(NORMALIZED_DATA_PATH).exists():
        print("=== STEP 1: Normalizing symptoms ===")
        stats = prepare_diagnostic_data(
            input_path=RAW_DATA_PATH,
            output_path=NORMALIZED_DATA_PATH,
            oov_report_path='/Users/robertdoherty/Desktop/Playground/research agent/output/2025-11-03/oov_report.txt'
        )
        print()
    else:
        print(f"=== Using existing normalized data: {NORMALIZED_DATA_PATH} ===\n")
    
    # Step 2: Load normalized data
    print("=== STEP 2: Loading data ===")
    data = load_data(NORMALIZED_DATA_PATH)
    print(f"Loaded {len(data)} samples\n")
    
    # Step 3: Create dataset
    dataset = DiagnosticPredictionDataset(data)
    
    # Split into train/val
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}\n")
    
    # Step 4: Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Step 5: Inspect data
    print("=== STEP 3: Inspecting first batch ===")
    sample_batch = next(iter(train_loader))
    print(f"Batch keys: {list(sample_batch.keys())}")
    print(f"Number of posts: {len(sample_batch['posts'])}")
    print(f"\nExample 1:")
    print(f"  Raw symptoms: {sample_batch['symptoms_raw'][0]}")
    print(f"  Normalized: {sample_batch['symptoms_normalized'][0]}")
    print(f"  Label: {sample_batch['labels'][0]}")
    print(f"  Post preview: {sample_batch['posts'][0][:150]}...")
    
    # Get unique labels
    print("\n=== STEP 4: Analyzing labels ===")
    all_labels = set()
    all_symptoms = set()
    for batch in train_loader:
        all_labels.update(batch['labels'])
        for symptom_list in batch['symptoms_normalized']:
            all_symptoms.update(symptom_list)
    
    print(f"Unique diagnostic labels: {len(all_labels)}")
    print(f"Labels: {sorted(all_labels)}")
    print(f"\nUnique normalized symptoms: {len(all_symptoms)}")
    print(f"Top symptoms: {sorted(list(all_symptoms))[:20]}")
    
    print("\n=== Ready for model development! ===")
    print("Next steps:")
    print("1. Implement text encoder (e.g., sentence-transformers for symptoms + posts)")
    print("2. Create multi-label classifier (one symptom → multiple possible diagnostics)")
    print("3. Implement training loop with proper metrics (accuracy, F1, etc.)")
    print("4. Add validation and evaluation")
    print("\nData is normalized and ready to use!")


if __name__ == "__main__":
    main()

