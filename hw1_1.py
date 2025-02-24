import os
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model_1 import PosePredictorMLP
from utils import *


class Config:
    max_buffer_length = 5
    initial_lr = 1e-3
    epochs = 1000
    data_per_epoch = 1
    batch_size = 1
    
    early_stop_patience = 110000
    early_stop_min_delta = 1e-5
    
    checkpoint_frequency = 1
    base_path = "model_checkpoints"
    
class EarlyStopping:
    def __init__(self, patience=Config.early_stop_patience, min_delta=Config.early_stop_min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class ModelSaver:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = os.path.join(Config.base_path, self.timestamp)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, epoch, model, optimizer, loss, is_best=False):
        """Save model checkpoint with timestamp"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        if is_best:
            filename = os.path.join(self.checkpoint_dir, f'best_model.pt')
        else:
            filename = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            # print("xdd")
            
        # torch.save(checkpoint, filename)
        print(f"Saved checkpoint to {filename}")
        
        metadata_file = os.path.join(self.checkpoint_dir, 'training_metadata.txt')
        with open(metadata_file, 'a') as f:
            f.write(f"Epoch {epoch}: Loss = {loss:.6f}, Timestamp = {checkpoint['timestamp']}\n")

def create_batch_from_buffer(buffer, indices):
    """Helper function to create batches from buffer samples"""
    batch_positions = torch.cat([buffer[idx][0] for idx in indices])
    batch_actions = torch.cat([buffer[idx][1] for idx in indices])
    batch_imgs_before = torch.cat([buffer[idx][2] for idx in indices])
    batch_imgs_after = torch.cat([buffer[idx][3] for idx in indices])
    return batch_positions, batch_actions, batch_imgs_before, batch_imgs_after

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = PosePredictorMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.initial_lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)
    criterion = torch.nn.MSELoss()
    early_stopping = EarlyStopping()
    model_saver = ModelSaver()
    
    buffer = deque(maxlen=Config.max_buffer_length)
    best_loss = float('inf')

    try:
        for epoch in range(Config.epochs):
            model.train()
            epoch_loss = 0
            
            positions, actions, imgs_before, imgs_after = collect(
                epoch % Config.data_per_epoch, 
                Config.data_per_epoch
            )
            imgs_before = imgs_before.float().to(device) / 255.0
            positions = positions.float().to(device)
            actions = actions.long().to(device)

            for i in range(Config.data_per_epoch):
                buffer.append((
                    positions[i:i+1],
                    actions[i:i+1],
                    imgs_before[i:i+1],
                    imgs_after[i:i+1]
                ))
            
            print(f"Buffer size: {len(buffer)}")
            
            if len(buffer) > Config.batch_size:
                num_batches = len(buffer) // Config.batch_size
                
                for batch in range(num_batches):
                    indices = np.random.choice(
                        len(buffer), 
                        size=Config.batch_size, 
                        replace=False
                    )
                    
                    batch_positions, batch_actions, batch_imgs_before, batch_imgs_after = create_batch_from_buffer(
                        buffer, indices
                    )
                    
                    predictions = model(batch_imgs_before, batch_actions)
                    loss = criterion(predictions, batch_positions)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    print(f"Epoch {epoch}, Batch {batch}/{num_batches}, Loss: {loss.item():.6f}")
            
            avg_loss = epoch_loss / num_batches if len(buffer) > Config.batch_size else float('inf')
            
            scheduler.step(avg_loss)
            
            early_stopping(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                model_saver.save_checkpoint(epoch, model, optimizer, avg_loss, is_best=True)
            
            print(f"Epoch {epoch} complete, Average Loss: {avg_loss:.6f}, "
                  f"Buffer size: {len(buffer)}, "
                  f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

            if (epoch + 1) % Config.checkpoint_frequency == 0:
                model_saver.save_checkpoint(epoch, model, optimizer, avg_loss)
                
            if early_stopping.should_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final checkpoint...")
        model_saver.save_checkpoint(epoch, model, optimizer, avg_loss, is_best=False)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        model_saver.save_checkpoint(epoch, model, optimizer, avg_loss, is_best=False)
        raise e
def test_model(model_path, grid_size=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PosePredictorMLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True)['model_state_dict'])
    model.eval()
    
    plt.figure(figsize=(20, 4))
    
    all_positions = []
    all_predictions = []
    all_actions = []

    for x in range(1, grid_size + 1):
        positions, actions, imgs_before, _ = collect(x-1, 1)
        imgs_before = imgs_before.float().to(device) / 255.0
        actions = actions.long().to(device)
        
        with torch.no_grad():
            predictions = model(imgs_before, actions).cpu().numpy()
        real_positions = positions.numpy()
        
        all_positions.append(real_positions[0])
        all_predictions.append(predictions[0])
        all_actions.append(actions.cpu().numpy()[0])
    
    all_positions = np.array(all_positions)
    all_predictions = np.array(all_predictions)
    all_actions = np.array(all_actions)
    for i in range(grid_size):
        plt.subplot(1, 5, i+1)
        
        real_pos = all_positions[i]
        pred_pos = all_predictions[i]
        action = all_actions[i]
        
        plt.scatter(real_pos[0], real_pos[1], c='blue', label='Real', s=100)
        plt.scatter(pred_pos[0], pred_pos[1], c='red', label='Predicted', s=100)
        plt.plot([real_pos[0], pred_pos[0]], 
                 [real_pos[1], pred_pos[1]], 
                 'k-', alpha=0.5)
        
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.xticks(range(-5, 6))
        plt.yticks(range(-5, 6))
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.grid(True)
        
        info_text = f'Action: {action}\n'
        info_text += f'Real: ({real_pos[0]:.2f}, {real_pos[1]:.2f})\n'
        info_text += f'Pred: ({pred_pos[0]:.2f}, {pred_pos[1]:.2f})'
        plt.title(f'Position {i+1}')
        plt.text(-4.8, -4.5, info_text, 
                bbox=dict(facecolor='white', alpha=0.8),
                fontsize=8)
        
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\nDetailed Results:")
    print("=" * 50)
    for i in range(grid_size):
        print(f"\nSample {i+1}:")
        print(f"Action: {all_actions[i]}")
        print(f"Real Position: ({all_positions[i][0]:.3f}, {all_positions[i][1]:.3f})")
        print(f"Predicted Position: ({all_predictions[i][0]:.3f}, {all_predictions[i][1]:.3f})")
        error = np.sqrt(((all_predictions[i] - all_positions[i]) ** 2).sum())
        print(f"Error: {error:.3f}")
    
    print("\nOverall Statistics:")
    print("=" * 50)
    mse = ((all_predictions - all_positions) ** 2).mean()
    mae = np.abs(all_predictions - all_positions).mean()
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")

if __name__ == "__main__":
    train()
    # test_model("hw1_1.pt", grid_size=5)