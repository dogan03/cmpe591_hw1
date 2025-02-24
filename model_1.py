import torch
import torch.nn as nn


class PosePredictorMLP(nn.Module):
    def __init__(self):  
        super(PosePredictorMLP, self).__init__()
        img_size = 3 * 128 * 128
        num_actions = 4  
        action_embedding_dim = 64          
        self.flatten = torch.nn.Flatten()
        
        self.action_embedding = nn.Embedding(num_actions, action_embedding_dim)
        
        img_output_size = 64
        combined_size = img_output_size + action_embedding_dim
        self.mlp_img = torch.nn.Sequential(
            torch.nn.Linear(img_size, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, img_output_size),
        )

        self.mlp_together = torch.nn.Sequential(
            torch.nn.Linear(combined_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        )

    def forward(self, img, action):
        x_img = self.flatten(img)
        
        x_action = self.action_embedding(action).squeeze(1)
        
        x_img = self.mlp_img(x_img)

        x = torch.cat([x_img, x_action], dim=1)
        
        x = self.mlp_together(x)
        return x