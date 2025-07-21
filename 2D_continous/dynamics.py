from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import wandb


class Trainer:
    def __init__(self, flow_model, score_model, shape:tuple, dataset, 
                use_wandb:bool=False,
                device='cuda', num_samples=2, lr=1e-3, weight_decay=1e-4, alpha=0.8, warmup_steps=500):
        self._check_flow_model(flow_model)
        self.flow_model = Wrapper2D(flow_model, shape).to(device)
        self.score_model = Wrapper2D(score_model, shape).to(device)
        self.device = device
        self.num_samples = num_samples
        self.dataset = dataset
        self.diffuser = Diffuser(alpha=alpha)

        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.use_wandb = use_wandb

        self.dim = 1
        for s in shape:
            self.dim *= s
    
    def _check_flow_model(self, flow_model):
        # ensure there is no ReLU or other non-2rd-order-differentiable operations
        for layer in flow_model.modules():
            if isinstance(layer, nn.ReLU):
                raise ValueError("ReLU activations don't support 2nd-order differentiation. Use SiLU or other differentiable activations instead.")
    
    def estimate(self, x):
        x = x.reshape(x.shape[0], -1)
        return div_estimate(self.flow_model, self.score_model, x, num_samples=self.num_samples)
    
    def setup_wandb(self):
        if self.use_wandb:
            wandb.init(project='pattern_to_dynamics', config={
                'lr': self.lr,
                'weight_decay': self.weight_decay,
                'alpha': self.diffuser.alpha,
                'num_samples': self.num_samples,
            })
    
    def train(self, epochs=10, batch_size=32):
        self.setup_wandb()
        optimizer = torch.optim.AdamW(self.flow_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=epochs * len(dataloader)
        )
        self.flow_model.train()

        lf = nn.MSELoss()
        
        for epoch in range(epochs):
            i = 0
            for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{epochs}"):
                scheduler.step()
                batch = batch.to(self.device)
                batch = self.diffuser.add_noise(batch, torch.randn_like(batch))
                optimizer.zero_grad()
                
                output = self.estimate(batch)
                loss = lf(output['div(pv)'], torch.zeros_like(output['div(pv)'])) / self.dim
                loss.backward()

                if self.use_wandb:
                    wandb.log({'loss': loss.item(), 'epoch': epoch+i/len(dataloader), 'lr': optimizer.param_groups[0]['lr']})
                
                optimizer.step()
                i += 1
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
        
        if self.use_wandb:
            wandb.finish()