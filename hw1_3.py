import os
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model_3 import ImagePredictor
from utils import *


class Config:
    def __init__(self):
        self.max_buffer_length = 128
        self.initial_buffer_size = 5
        self.data_per_epoch = 1
        self.batch_size = 1
        self.epochs = 10000
        self.initial_lr = 1e-5
        self.step_size = 10
        self.weight_decay = 1e-5
        self.adam_betas = (0.9, 0.999)
        self.grad_clip = 1.0
        self.lr_factor = 0.5
        self.lr_patience = 15
        self.lr_min = 1e-6
        self.lr_cooldown = 5
        self.early_stop_patience = 30
        self.early_stop_min_delta = 1e-6
        
        self.checkpoint_frequency = 10  # Save every 10 epochs
        self.base_path = "model_checkpoints"
        self.save_best = True          # Save best model


class EarlyStopping:
    """Early stopping implementation"""
    def __init__(self, config):
        self.patience = config.early_stop_patience
        self.min_delta = config.early_stop_min_delta
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
    """Handles model checkpointing"""
    def __init__(self, config):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = os.path.join(config.base_path, self.timestamp)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, epoch, model, optimizer, loss, global_step, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'global_step': global_step,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        filename = os.path.join(
            self.checkpoint_dir, 
            'best_model.pt' if is_best else f'checkpoint_epoch_{epoch}.pt'
        )
        
        torch.save(checkpoint, filename)
        print(f"Saved checkpoint to {filename}")
        
        # Save metadata
        with open(os.path.join(self.checkpoint_dir, 'training_metadata.txt'), 'a') as f:
            f.write(f"Epoch {epoch}: Loss = {loss:.6f}, Timestamp = {checkpoint['timestamp']}\n")


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ImagePredictor().to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.initial_lr,
            betas=config.adam_betas,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.lr_factor,
            patience=config.lr_patience,
            min_lr=config.lr_min,
            cooldown=config.lr_cooldown,
            verbose=True
        )
        
        self.early_stopping = EarlyStopping(config)
        self.model_saver = ModelSaver(config)
        self.buffer = deque(maxlen=config.max_buffer_length)
        
        self.best_loss = float('inf')
        self.best_epoch = 0

    def create_batch_from_buffer(self, indices):
        """Create a batch from buffer using given indices"""
        batch_actions = torch.cat([self.buffer[idx][0] for idx in indices])
        batch_imgs_before = torch.cat([self.buffer[idx][1] for idx in indices])
        batch_imgs_after = torch.cat([self.buffer[idx][2] for idx in indices])
        return batch_actions, batch_imgs_before, batch_imgs_after
        
    def train(self):
        global_step = 0
        
        try:
            for epoch in range(self.config.epochs):
                self.model.train()
                epoch_loss = self.train_epoch(epoch, global_step)
                
                if epoch_loss < self.best_loss:
                    self.best_loss = epoch_loss
                    self.best_epoch = epoch
                    if self.config.save_best:
                        self.model_saver.save_checkpoint(
                            epoch, self.model, self.optimizer, 
                            epoch_loss, global_step, is_best=True
                        )
                
                if epoch % self.config.checkpoint_frequency == 0:
                    self.model_saver.save_checkpoint(
                        epoch, self.model, self.optimizer,
                        epoch_loss, global_step, is_best=False
                    )
                
                self.scheduler.step(epoch_loss)
                
                self.early_stopping(epoch_loss)
                if self.early_stopping.should_stop:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
                
                print(f"\nEpoch {epoch}:")
                print(f"Current Loss: {epoch_loss:.6f}")
                print(f"Best Loss: {self.best_loss:.6f} (Epoch {self.best_epoch})")
                print(f"Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise e
        finally:
            self.model_saver.save_checkpoint(
                epoch, self.model, self.optimizer, epoch_loss,
                global_step, is_best=False
            )
            
            if self.best_epoch != epoch:
                self.model_saver.save_checkpoint(
                    self.best_epoch, self.model, self.optimizer,
                    self.best_loss, global_step, is_best=True
                )

    def train_epoch(self, epoch, global_step):
        total_loss = 0
        num_batches = 0
        
        _, actions, imgs_before, imgs_after = collect(
            epoch % self.config.data_per_epoch,
            self.config.data_per_epoch
        )
        
        imgs_before = imgs_before.float().to(self.device) / 255.0
        imgs_after = imgs_after.float().to(self.device) / 255.0
        actions = actions.long().to(self.device)
        
        for i in range(self.config.data_per_epoch):
            self.buffer.append((
                actions[i:i+1],
                imgs_before[i:i+1],
                imgs_after[i:i+1]
            ))
        
        if len(self.buffer) <= self.config.initial_buffer_size:
            return float('inf')
        
        for step in range(self.config.step_size):
            indices = np.random.choice(
                len(self.buffer),
                size=self.config.batch_size,
                replace=False
            )
            
            batch_actions, batch_imgs_before, batch_imgs_after = \
                self.create_batch_from_buffer(indices)
            
            step_loss = 0
            for i in range(self.config.batch_size):
                single_img_before = batch_imgs_before[i:i+1]
                single_action = batch_actions[i:i+1]
                single_img_after = batch_imgs_after[i:i+1]
                
                predicted = self.model(single_img_before, single_action)
                
                loss = self.model.compute_loss(
                    single_img_before,
                    predicted,
                    single_action,
                    single_img_after
                )
                
                if epoch % 250000 == 0 and step == 0 and i == 0:
                    self.model.visualize(
                        single_img_before,
                        single_img_after,
                        single_action
                    )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
                self.optimizer.step()
                
                step_loss += loss.item()
            
            avg_step_loss = step_loss / self.config.batch_size
            total_loss += avg_step_loss
            num_batches += 1
            global_step += 1
            
            print(f"Step {step + 1}/{self.config.step_size}, "
                  f"Loss: {avg_step_loss:.6f}")
        
        return total_loss / num_batches
def test_model(model_path, num_samples=2):
    """Test the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImagePredictor().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)['model_state_dict']
    )
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i in range(num_samples):
        _, actions, imgs_before, imgs_after = collect(i, 1)
        
        imgs_before = imgs_before.float().to(device) / 255.0
        actions = actions.long().to(device)
        
        with torch.no_grad():
            predicted_img = model(imgs_before, actions)
        
        img_before = imgs_before[0].cpu().numpy().transpose(1, 2, 0)
        img_after = imgs_after[0].numpy().transpose(1, 2, 0) / 255.0
        img_predicted = predicted_img[0].cpu().numpy().transpose(1, 2, 0)
        
        axes[i, 0].imshow(img_before)
        axes[i, 0].set_title(f'Before (Action: {actions[0].item()})')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(img_predicted)
        axes[i, 1].set_title('Predicted')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(img_after)
        axes[i, 2].set_title('Actual')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    config = Config()
    # trainer = Trainer(config)
    # trainer.train()
    test_model("hw1_3.pt", num_samples=5)