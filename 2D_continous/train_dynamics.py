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


class SymConv2d_3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SymConv2d_3, self).__init__()
        self.register_buffer('mask_center', torch.Tensor(
            [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]]))
        
        self.register_buffer('mask_corner', torch.Tensor(
            [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]]))
        self.register_buffer('mask_cross', torch.Tensor(
            [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]]))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.padding = 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.center = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.corner = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.cross = nn.Parameter(torch.Tensor(out_channels, in_channels))
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.center)
        nn.init.xavier_uniform_(self.corner)
        nn.init.xavier_uniform_(self.cross)
        nn.init.zeros_(self.bias)
    
    @property
    def kernel(self):
        k = torch.einsum('oi, hw -> oihw', self.center, self.mask_center) + \
        torch.einsum('oi, hw -> oihw', self.corner, self.mask_corner) + \
        torch.einsum('oi, hw -> oihw', self.cross, self.mask_cross)
        return k
    
    def conv(self, x):
        return nn.functional.conv2d(
            x, self.kernel, bias=self.bias, padding=self.padding, groups=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class FlowModel(nn.Module):
    def __init__(self, hidden_channels=32):
        super(FlowModel, self).__init__()
        self.cnn = nn.Sequential(
            SymConv2d_3(2, hidden_channels),
            # VPJBatchNorm2d(hidden_channels, affine=False),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=0),
            # VPJBatchNorm2d(hidden_channels, affine=False),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 2, kernel_size=1, padding=0),
        )
        self.bn = VPJBatchNorm2d(2, affine=False)

        # self.skip_connection = nn.Conv2d(2, 2, kernel_size=1, padding=0)
    
    def forward(self, x):
        return self.bn(self.cnn(x))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a flow model for Turing patterns.')
    parser.add_argument('--dataset', type=str, default='spirals',
                        help='Dataset to use for training the flow model.')
    parser.add_argument('-a', '--alpha', type=float, default=0.8,
                        help='Alpha value for the diffusion model.')

    args = parser.parse_args()
    dataset = TuringPatternDataset.load(f'./turing_pattern/data/{args.dataset}_128x128.pt')
    score_model = load_score_model(args.dataset, device='cuda')
    flow_model = FlowModel().to('cuda')
    trainer = Trainer(
        flow_model, score_model, 
        (2, 128, 128), # It is very important to set the correct input shape
        dataset=dataset, 
        use_wandb=True, 
        lr=1e-3, 
        num_samples=8, 
        weight_decay=1e-5, 
        gradient_accumulation_steps=8,
        alpha=args.alpha)

    trainer.train(epochs=20, batch_size=128)
    # save flow_model
    import os
    os.makedirs('./turing_pattern/flow_models/', exist_ok=True)
    # save the model state dict
    torch.save(flow_model.state_dict(), f'./turing_pattern/flow_models/{args.dataset}.pth')