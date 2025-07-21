import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from probflow import Diffuser, ResNet2D, VPJBatchNorm2d, VPJBatchNorm, div_estimate
from dynamics import Trainer
from turing_pattern import GrayScottSimulator, create_random_state, TuringPatternDataset


def load_score_model(name:str, device:str='cuda', freeze:bool=True) -> nn.Module:
    import os
    model = ResNet2D(in_channels=2, block_channels=[16, 32, 64], block_rs=[1, 1, 2])
    path = os.path.join(f'./turing_pattern/score_models/{name}/', 'model.pth')
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model


class FlowModel(nn.Module):
    def __init__(self):
        super(FlowModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(32, 2, kernel_size=1, padding=0),
            VPJBatchNorm2d(2, affine=False),
        )
    
    def forward(self, x):
        return self.cnn(x)


if __name__ == "__main__":
    dataset = TuringPatternDataset.load(f'./turing_pattern/data/spirals_128x128.pt')
    score_model = load_score_model('spirals', device='cuda')
    flow_model = FlowModel().to('cuda')
    trainer = Trainer(
        flow_model, score_model, 
        (2, 64, 64), 
        dataset=dataset, use_wandb=True, lr=1e-3, num_samples=10)
    
    trainer.train(epochs=100, batch_size=128)
    # save flow_model
    import os
    os.makedirs('./turing_pattern/flow_models/', exist_ok=True)
    # save the model state dict
    torch.save(flow_model.state_dict(), './turing_pattern/flow_models/spirals.pth')