import numpy as np
import torch
from tqdm import tqdm

from homework_1 import Hw1Env


def collect(idx, N):
    env = Hw1Env(render_mode="offscreen")
    positions = torch.zeros(N, 2, dtype=torch.float)
    actions = torch.zeros(N, dtype=torch.uint8)
    imgs_before = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    imgs_after = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    for i in tqdm(range(N), desc=f"Collecting data for {idx}"):
        env.reset()
        action_id = np.random.randint(4)
        _, img_before = env.state()
        imgs_before[i] = img_before
        env.step(action_id)
        obj_pos, img_after = env.state()
        positions[i] = torch.tensor(obj_pos)
        actions[i] = action_id
        imgs_after[i] = img_after
    
    return positions, actions, imgs_before, imgs_after
