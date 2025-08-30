import torch
from torch.utils.data import DataLoader
from datasets.data_loader import load_gold_stream # <-- The ONLY data import needed

# --- 1. Load the Data Stream (Location Agnostic) ---
# This single line works for both local dev and production S3 streaming
streamable_dataset = load_gold_stream()

# --- 2. Prepare for PyTorch ---
streamable_dataset = streamable_dataset.with_format(type="torch", columns=["features", "label"])
shuffled_stream = streamable_dataset.shuffle(buffer_size=10000, seed=42)

# --- 3. Define Model and Training Loop (No changes here) ---
# ... (all your PyTorch model, optimizer, and training loop code) ...
# ... (it remains exactly the same as the previous version) ...

# Example of using the loader in the loop
train_loader = DataLoader(shuffled_stream, batch_size=32)
for batch in train_loader:
    # training logic...
    pass

print("Training complete.")