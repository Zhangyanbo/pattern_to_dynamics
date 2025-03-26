import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from distributions import LorenzDataset, RingDataset, TwoPeaksDataset, TwoMoonsDataset

## Model

class Diffusion(nn.Module):
    def __init__(self, num_steps=100, dim=3):
        super(Diffusion, self).__init__()
        self.nonlinear = nn.ReLU()
        self.num_steps = num_steps
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim+1, 64),
            self.nonlinear,
            nn.Linear(64, 64),
            self.nonlinear,
            nn.Linear(64, dim),
        )
        self.alpha = self.create_alpha(num_steps) # alpha_T --> 0, alpha_0 --> 1
    
    def create_alpha(self, num_steps):
        alpha_t = 1 - torch.linspace(1e-3, 1-1e-3, num_steps)
        # covert to embedding layer, set alpha as a buffer
        alpha = nn.Embedding(num_steps, 1)
        alpha.weight.data = alpha_t.reshape(-1, 1)
        # don't train the embedding layer
        alpha.weight.requires_grad = False
        return alpha
    
    def eps_predictor(self, x, t):
        input = torch.cat([x, t], dim=-1)
        return self.mlp(input) + x
    
    def score(self, x, t:float=0.1):
        if isinstance(t, float):
            batch_size = x.shape[0]
            t = torch.ones(batch_size).to(x.device) * t
        alpha_t = self.alpha_t(t)
        eps_pred = self.eps_predictor(x, t.unsqueeze(-1))
        return -eps_pred / (1 - alpha_t).sqrt()
    
    def forward(self, x, t):
        xt, eps = self.diffuse(x, t)
        eps_pred = self.eps_predictor(xt, t.unsqueeze(-1))
        return eps, eps_pred
    
    def alpha_t(self, t):
        t = (t * (self.num_steps - 1)).long()
        alpha_t = self.alpha(t)
        return alpha_t

    def diffuse(self, x, t):
        alpha_t = self.alpha_t(t)
        eps = torch.randn_like(x)
        xt = alpha_t.sqrt() * x + (1 - alpha_t).sqrt() * eps
        return xt, eps
    
    def sample(self, num_sample):
        x = torch.randn(num_sample, self.dim)
        # from t=self.num_steps to t=1, inverse diffusion process
        for i in reversed(range(2, self.num_steps)):
            t = torch.ones(num_sample) * (i / self.num_steps)
            T = (t * (self.num_steps - 1)).long()
            alpha_T = self.alpha(T)
            alpha_T_1 = self.alpha(T-1)
            with torch.no_grad():
                eps_pred = self.eps_predictor(x, t.unsqueeze(-1))
            x0_pred = (x - (1-alpha_T).sqrt() * eps_pred) / alpha_T.sqrt()
            x = alpha_T_1.sqrt() * x0_pred + (1-alpha_T_1).sqrt() * eps_pred
        return x


def train(dataset, num_epochs=500):
    from tqdm import tqdm
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = Diffusion(dim=dataset.dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    losses = []
    lf = nn.MSELoss()

    for epoch in tqdm(range(num_epochs)):
        acc_loss = 0
        for x in dataloader:
            optimizer.zero_grad()
            t = torch.rand(x.shape[0])
            eps, eps_pred = model.forward(x, t)
            loss = lf(eps, eps_pred)
            loss.backward()
            optimizer.step()
            acc_loss += loss.item()
        losses.append(acc_loss / len(dataloader))
    
    model.eval()

    return model, losses

def test_sample(model, dataset, args, num_sample=100):
    sampled = model.sample(num_sample)
    if args.model == "lorenz":
        marker = '-'
    else:
        marker = '.'
    plt.plot(dataset.time_series[:, 0], dataset.time_series[:, 1], marker, alpha=0.5, label="True")
    plt.plot(sampled[:, 0], sampled[:, 1], '.', alpha=0.5, label="Sampled")
    plt.title("Sampled")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(f"./figure/{args.model}_sampled.png")
    plt.close()

def plot_losses(losses, args):
    plt.plot(losses)
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"./figure/{args.model}_losses.png")
    plt.close()

def save_model(model, args):
    torch.save(model.state_dict(), f'./models/{args.model}_ddim.pth')
    print(f"Model saved to ./models/{args.model}_ddim.pth")

if __name__ == "__main__":
    import argparse
    # set model: choose from lorenz, ring, two_peaks
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lorenz")
    parser.add_argument("--num_epochs", type=int, default=512)
    parser.add_argument("--num_sample", type=int, default=1000)
    args = parser.parse_args()
    if args.model == "lorenz":
        dataset = LorenzDataset()
    elif args.model == "ring":
        dataset = RingDataset()
    elif args.model == "two_peaks":
        dataset = TwoPeaksDataset()
    elif args.model == "two_moons":
        dataset = TwoMoonsDataset()
    else:
        raise ValueError(f"Model {args.model} not supported")

    model, losses = train(dataset, num_epochs=args.num_epochs)
    plot_losses(losses, args)
    test_sample(model, dataset, args, num_sample=args.num_sample)
    save_model(model, args)