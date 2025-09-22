import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from net import DecoderTransformer


def get_lr_schedule(optimizer, step, warmup_steps=2000, lr=6e-4):
    """Learning rate schedule with warmup"""
    if step < warmup_steps:
        # Linear warmup
        lr_mult = step / warmup_steps
    else:
        # Constant after warmup
        lr_mult = 1.0
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * lr_mult
    
    return lr * lr_mult


def train_step(model, batch, optimizer, criterion, max_grad_norm=1.0):
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    
    # Assuming batch is (input_ids, target_ids)
    inputs, targets = batch
    
    # Forward pass
    outputs = model(inputs)
    
    # Reshape for loss calculation
    # outputs: (batch_size, seq_len, vocab_size)
    # targets: (batch_size, seq_len)
    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    # Optimizer step
    optimizer.step()
    
    return loss.item()


def train_model(model, train_dataloader, num_epochs, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Main training loop"""
    
    # Move model to device
    model = model.to(device)
    
    # Optimizer with specified parameters
    optimizer = optim.Adam(
        model.parameters(),
        lr=6e-4,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Assuming -1 for padding
    
    # Training loop
    step = 0
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            else:
                batch = batch.to(device)
            
            # Update learning rate with warmup
            current_lr = get_lr_schedule(optimizer, step, warmup_steps=2000, lr=6e-4)
            
            # Training step
            loss = train_step(model, batch, optimizer, criterion, max_grad_norm=1.0)
            
            epoch_loss += loss
            num_batches += 1
            step += 1
            
            # Log progress
            if step % 100 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss:.4f}, LR: {current_lr:.6f}")
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
    
    return model


def create_model(vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=3072, max_seq_length=5000, dropout=0.1):
    """Create and return the model"""
    model = DecoderTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout
    )
    return model


if __name__ == "__main__":
    # Example usage
    vocab_size = 50000  # Adjust based on your vocabulary
    
    # Create model
    model = create_model(vocab_size)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # You would need to provide your own dataloader here
    # train_dataloader = DataLoader(your_dataset, batch_size=32, shuffle=True)
    # trained_model = train_model(model, train_dataloader, num_epochs=10)