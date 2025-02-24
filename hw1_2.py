import os
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model_1 import PosePredictorMLP
from model_2 import PosePredictorCNN
from utils import *


class Config:
    max_buffer_length = 5
    initial_lr = 1e-3
    epochs = 1000
    data_per_epoch = 1
    batch_size = 1
    
    early_stop_patience = 10000
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
    
    model = PosePredictorCNN().to(device)
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
def test_models(mlp_model_path, cnn_model_path, grid_size=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mlp_model = PosePredictorMLP().to(device)
    cnn_model = PosePredictorCNN().to(device)
    
    mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=device, weights_only=True)['model_state_dict'])
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device, weights_only=True)['model_state_dict'])
    
    mlp_model.eval()
    cnn_model.eval()
    
    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(25, 10))
    regular_axes = [ax1, ax2, ax3, ax4, ax5]
    zoomed_axes = [ax6, ax7, ax8, ax9, ax10]
    
    all_positions = []
    mlp_predictions = []
    cnn_predictions = []
    all_actions = []
    
    for x in range(1, grid_size + 1):
        positions, actions, imgs_before, _ = collect(x-1, 1)
        imgs_before = imgs_before.float().to(device) / 255.0
        actions = actions.long().to(device)
        
        with torch.no_grad():
            mlp_preds = mlp_model(imgs_before, actions).cpu().numpy()
            cnn_preds = cnn_model(imgs_before, actions).cpu().numpy()
        real_positions = positions.numpy()
        
        all_positions.append(real_positions[0])
        mlp_predictions.append(mlp_preds[0])
        cnn_predictions.append(cnn_preds[0])
        all_actions.append(actions.cpu().numpy()[0])
    
    all_positions = np.array(all_positions)
    mlp_predictions = np.array(mlp_predictions)
    cnn_predictions = np.array(cnn_predictions)
    all_actions = np.array(all_actions)
    
    for i in range(grid_size):
        regular_axes[i].scatter(all_positions[i, 0], all_positions[i, 1], c='blue', label='Real', s=100)
        regular_axes[i].scatter(mlp_predictions[i, 0], mlp_predictions[i, 1], c='red', label='MLP', s=100)
        regular_axes[i].scatter(cnn_predictions[i, 0], cnn_predictions[i, 1], c='green', label='CNN', s=100)
        
        regular_axes[i].plot([all_positions[i, 0], mlp_predictions[i, 0]], 
                           [all_positions[i, 1], mlp_predictions[i, 1]], 
                           'r--', alpha=0.3)
        regular_axes[i].plot([all_positions[i, 0], cnn_predictions[i, 0]], 
                           [all_positions[i, 1], cnn_predictions[i, 1]], 
                           'g--', alpha=0.3)
        
        regular_axes[i].set_xlim(-5, 5)
        regular_axes[i].set_ylim(-5, 5)
        regular_axes[i].set_xticks(range(-5, 6))
        regular_axes[i].set_yticks(range(-5, 6))
        regular_axes[i].grid(True)
        regular_axes[i].set_title(f'Position {i+1} - Full Scale')
        
        margin = 0.2
        min_x = min(all_positions[i, 0], mlp_predictions[i, 0], cnn_predictions[i, 0]) - margin
        max_x = max(all_positions[i, 0], mlp_predictions[i, 0], cnn_predictions[i, 0]) + margin
        min_y = min(all_positions[i, 1], mlp_predictions[i, 1], cnn_predictions[i, 1]) - margin
        max_y = max(all_positions[i, 1], mlp_predictions[i, 1], cnn_predictions[i, 1]) + margin
        
        zoomed_axes[i].scatter(all_positions[i, 0], all_positions[i, 1], c='blue', label='Real', s=100)
        zoomed_axes[i].scatter(mlp_predictions[i, 0], mlp_predictions[i, 1], c='red', label='MLP', s=100)
        zoomed_axes[i].scatter(cnn_predictions[i, 0], cnn_predictions[i, 1], c='green', label='CNN', s=100)
        
        zoomed_axes[i].plot([all_positions[i, 0], mlp_predictions[i, 0]], 
                          [all_positions[i, 1], mlp_predictions[i, 1]], 
                          'r--', alpha=0.3)
        zoomed_axes[i].plot([all_positions[i, 0], cnn_predictions[i, 0]], 
                          [all_positions[i, 1], cnn_predictions[i, 1]], 
                          'g--', alpha=0.3)
        
        zoomed_axes[i].set_xlim(min_x, max_x)
        zoomed_axes[i].set_ylim(min_y, max_y)
        zoomed_axes[i].grid(True)
        zoomed_axes[i].set_title(f'Position {i+1} - Zoomed')
        
        info_text = f'Action: {all_actions[i]}\n'
        info_text += f'Real: ({all_positions[i,0]:.2f}, {all_positions[i,1]:.2f})\n'
        info_text += f'MLP: ({mlp_predictions[i,0]:.2f}, {mlp_predictions[i,1]:.2f})\n'
        info_text += f'CNN: ({cnn_predictions[i,0]:.2f}, {cnn_predictions[i,1]:.2f})'
        
        zoomed_axes[i].text(0.02, 0.02, info_text,
                          transform=zoomed_axes[i].transAxes,
                          bbox=dict(facecolor='white', alpha=0.8),
                          fontsize=8,
                          verticalalignment='bottom')
        
        if i == 0:  
            regular_axes[i].legend()
            zoomed_axes[i].legend()
    
    plt.tight_layout()
    plt.show()
    
    print("\nDetailed Results:")
    print("=" * 50)
    for i in range(grid_size):
        print(f"\nPosition {i+1}:")
        print(f"Action: {all_actions[i]}")
        print(f"Real Position: ({all_positions[i,0]:.3f}, {all_positions[i,1]:.3f})")
        print(f"MLP Prediction: ({mlp_predictions[i,0]:.3f}, {mlp_predictions[i,1]:.3f})")
        print(f"CNN Prediction: ({cnn_predictions[i,0]:.3f}, {cnn_predictions[i,1]:.3f})")
        
        mlp_error = np.sqrt(((mlp_predictions[i] - all_positions[i]) ** 2).sum())
        cnn_error = np.sqrt(((cnn_predictions[i] - all_positions[i]) ** 2).sum())
        print(f"MLP Error: {mlp_error:.3f}")
        print(f"CNN Error: {cnn_error:.3f}")
    
    print("\nOverall Statistics:")
    print("=" * 50)
    mlp_mse = ((mlp_predictions - all_positions) ** 2).mean()
    mlp_mae = np.abs(mlp_predictions - all_positions).mean()
    cnn_mse = ((cnn_predictions - all_positions) ** 2).mean()
    cnn_mae = np.abs(cnn_predictions - all_positions).mean()
    
    print("MLP Metrics:")
    print(f"Mean Squared Error: {mlp_mse:.6f}")
    print(f"Mean Absolute Error: {mlp_mae:.6f}")
    
    print("\nCNN Metrics:")
    print(f"Mean Squared Error: {cnn_mse:.6f}")
    print(f"Mean Absolute Error: {cnn_mae:.6f}")

if __name__ == "__main__":
    train()
    # test_models(mlp_model_path="hw1_1.pt", cnn_model_path="hw1_2.pt", grid_size=5) 