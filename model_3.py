import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_dilation


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ActionEmbedding(nn.Module):
    def __init__(self, num_actions=4, embedding_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, embedding_dim)
        self.transform = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, action):
        x = self.embedding(action)
        return self.transform(x)

class Encoder(nn.Module):
    def __init__(self, action_dim=256):
        super().__init__()
        
        self.init_features = nn.Sequential(
            nn.Conv2d(3, 64, 7, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 64)
        )
        
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        
        self.stage2 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )
        
        self.stage3 = nn.Sequential(
            ResidualBlock(256, 512, stride=2),
            ResidualBlock(512, 512)
        )
        
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Sigmoid()
        )
        
        self.attention = nn.Sequential(
            nn.Conv2d(512, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, action_emb):
        x0 = self.init_features(x)       
        
        x1 = self.stage1(x0)             
        x2 = self.stage2(x1)             
        x3 = self.stage3(x2)             
        
        action_feat = self.action_proj(action_emb)
        action_feat = action_feat.view(*action_feat.shape, 1, 1)
        
        attention = self.attention(x3)
        x3 = x3 * attention * action_feat.expand_as(x3)
        
        return x3, [x0, x1, x2]

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.refine3 = nn.Sequential(
            ResidualBlock(512, 256)  
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.refine2 = nn.Sequential(
            ResidualBlock(256, 128)  
        )
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.refine1 = nn.Sequential(
            ResidualBlock(128, 64)  
        )
        
        # Final layers
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x, skip_features):
        x0, x1, x2 = skip_features
        
        x = self.up3(x)                    
        x = torch.cat([x, x2], dim=1)      
        x = self.refine3(x)                
        
        x = self.up2(x)                    
        x = torch.cat([x, x1], dim=1)     
        x = self.refine2(x)               
        
        x = self.up1(x)                    
        x = torch.cat([x, x0], dim=1)   
        x = self.refine1(x)              
        
        x = self.final(x)
        
        return x

class ImagePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_embedding = ActionEmbedding()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x, action):
        action_emb = self.action_embedding(action)
        features, skip_features = self.encoder(x, action_emb)
        prediction = self.decoder(features, skip_features)
        return prediction
    
    def compute_loss(self, input_img, prediction, action, target):
        def get_red_mask(img):
            r = img[:, 0, :, :]
            g = img[:, 1, :, :]
            b = img[:, 2, :, :]
            avg_gb = (g + b) / 2
            return (r > avg_gb).float()

        red_mask = get_red_mask(target)
        
        weights = 1.0 + 9.0 * red_mask.unsqueeze(1)  
        
        loss = F.mse_loss(prediction, target, reduction='none')
        weighted_loss = (loss * weights).mean()

        return weighted_loss
    def visualize(self, input_img, target_img, action):
        with torch.no_grad():
            prediction = self(input_img, action)
            
            def get_red_mask(img):
                r = img[:, 0, :, :]
                g = img[:, 1, :, :]
                b = img[:, 2, :, :]
                avg_gb = (g + b) / 2
                red_mask = (r > avg_gb).float()
                
                red_mask_np = red_mask.cpu().numpy()
                kernel = np.ones((5, 5))
                
                dilated_masks = []
                for mask in red_mask_np:
                    dilated = binary_dilation(mask, kernel)
                    dilated_masks.append(dilated)
                
                return torch.from_numpy(np.stack(dilated_masks)).to(img.device)
            
            target_red_mask = get_red_mask(target_img)
            pred_red_mask = get_red_mask(prediction)
            
            input_np = input_img[0].cpu().permute(1,2,0).numpy()
            pred_np = prediction[0].cpu().permute(1,2,0).numpy()
            target_np = target_img[0].cpu().permute(1,2,0).numpy()
            
            error_map = np.abs(pred_np - target_np).mean(axis=2)
            red_error_map = error_map * target_red_mask[0].cpu().numpy()
            
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            
            axes[0,0].imshow(input_np)
            axes[0,0].set_title(f'Input (Action: {action.item()})')
            axes[0,0].axis('off')
            
            axes[0,1].imshow(pred_np)
            axes[0,1].set_title('Predicted')
            axes[0,1].axis('off')
            
            axes[0,2].imshow(target_np)
            axes[0,2].set_title('Target')
            axes[0,2].axis('off')
            
            axes[1,0].imshow(target_red_mask[0].cpu(), cmap='hot')
            axes[1,0].set_title('Target Red Regions')
            axes[1,0].axis('off')
            
            axes[1,1].imshow(pred_red_mask[0].cpu(), cmap='hot')
            axes[1,1].set_title('Predicted Red Regions')
            axes[1,1].axis('off')
            
            mask_diff = (pred_red_mask[0] - target_red_mask[0]).cpu()
            axes[1,2].imshow(mask_diff, cmap='bwr')  # blue-white-red colormap
            axes[1,2].set_title('Red Mask Difference\n(Blue: Missing, Red: Extra)')
            axes[1,2].axis('off')
            
            axes[2,0].imshow(error_map, cmap='viridis')
            axes[2,0].set_title('Overall Error Map')
            axes[2,0].axis('off')
            
            axes[2,1].imshow(red_error_map, cmap='viridis')
            axes[2,1].set_title('Red Region Error Map')
            axes[2,1].axis('off')
            
            axes[2,2].hist(pred_np[:,:,0].flatten(), bins=50, alpha=0.5, color='red', label='Pred R')
            axes[2,2].hist(pred_np[:,:,1].flatten(), bins=50, alpha=0.5, color='green', label='Pred G')
            axes[2,2].hist(pred_np[:,:,2].flatten(), bins=50, alpha=0.5, color='blue', label='Pred B')
            axes[2,2].set_title('Color Distribution')
            axes[2,2].legend()
            
            mse = F.mse_loss(prediction, target_img).item()
            red_mse = F.mse_loss(
                prediction * target_red_mask.unsqueeze(1),
                target_img * target_red_mask.unsqueeze(1)
            ).item()
            
            metrics_text = (
                f'MSE: {mse:.6f}\n'
                f'Red Region MSE: {red_mse:.6f}\n'
                f'Target Red Coverage: {target_red_mask.mean().item():.2%}\n'
                f'Predicted Red Coverage: {pred_red_mask.mean().item():.2%}'
            )
            
            plt.figtext(
                0.02, 0.02, metrics_text,
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8)
            )
            
            plt.tight_layout()
            plt.show()