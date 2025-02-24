import torch
import torch.nn as nn
import torch.nn.functional as F


class RedAttention(nn.Module):
    def __init__(self):
        super(RedAttention, self).__init__()
        self.conv_red = nn.Conv2d(3, 1, kernel_size=1)
        with torch.no_grad():
            self.conv_red.weight.data = torch.tensor([[1.0, -0.5, -0.5]]).view(1, 3, 1, 1)
            self.conv_red.bias.data = torch.tensor([0.0])

class PosePredictorCNN(nn.Module):
    def __init__(self):  
        super(PosePredictorCNN, self).__init__()
        num_actions = 4
        action_embedding_dim = 64
        
        self.action_embedding = nn.Embedding(num_actions, action_embedding_dim)
        
        self.red_attention = RedAttention()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.cnn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
        )
        
        combined_size = 64 + action_embedding_dim
        self.mlp_together = nn.Sequential(
            nn.Linear(combined_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, img, action):
        red_mask = self.red_attention.conv_red(img)
        red_mask = torch.sigmoid(red_mask)  
        
        x_img = torch.cat([img, red_mask], dim=1)  
        
        x_img = self.cnn(x_img)
        x_img = self.cnn_fc(x_img)
        
        x_action = self.action_embedding(action).squeeze(1)
        
        x = torch.cat([x_img, x_action], dim=1)
        x = self.mlp_together(x)
        return x
