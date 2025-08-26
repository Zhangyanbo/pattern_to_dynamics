from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import wandb
import torch
import torch.nn as nn
from probflow import Diffuser, div_estimate
import torch.nn.functional as F


class Wrapper2D(nn.Module):
    # convert vector input into [batch, *shape]
    def __init__(self, model: nn.Module, shape: tuple, gaussian_weight: float = 0.0, timestep: int=None):
        """
        A wrapper for 2D models to handle vector inputs.

        Args:
            model (nn.Module): The model to wrap.
            shape (tuple): The shape of the input tensor (excluding batch size).
            gaussian_weight (float): Weight for the Gaussian term in the score model.
        """
        super(Wrapper2D, self).__init__()
        self.model = model
        self.shape = shape
        self.gaussian_weight = gaussian_weight
        self.timestep = timestep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prefix_shape = x.shape[:-1]
        x = x.reshape(-1, *self.shape)
        if self.timestep is None:
            y = self.model(x)
        else:
            y = self.model(x, timestep=self.timestep)
        
        if not isinstance(y, torch.Tensor):
            y = y.sample

        if self.gaussian_weight > 0:
            gaussian_term = self.gaussian_weight * x
            y = y + gaussian_term
        return y.reshape(*prefix_shape, -1)


def score_function(unet, scheduler, t):
    def score_model(xt):
        model_output = unet(xt, t)
        x0 = scheduler.step(model_output.sample, t, xt).pred_original_sample
        alpha_t = scheduler.alphas_cumprod[t]
        score = (x0 * alpha_t ** 0.5 - xt) / (1 - alpha_t)
        return score

    return score_model


class Trainer:
    def __init__(self, 
                flow_model, 
                score_model, 
                scheduler,
                shape:tuple, 
                dataset, 
                use_wandb:bool=False,
                reference_flow_model:callable=None,
                model:str='none',
                device='cuda', 
                num_samples=2, 
                diffuse_time_t=10,
                lr=1e-3, 
                weight_decay=1e-4, 
                alpha=0.8, 
                gradient_accumulation_steps=1,
                schedule:bool=False,
                gaussian_weight=0.0,
                symmetry_punalty:float=0.0,
                warmup_steps=500
                ):
        self._check_flow_model(flow_model)
        self.flow_model = Wrapper2D(flow_model, shape).to(device)
        self.score_model = Wrapper2D(
                                score_function(score_model, scheduler, diffuse_time_t), 
                                shape).to(device)
        self.device = device
        self.num_samples = num_samples
        self.dataset = dataset
        self.diffuser = scheduler
        self.schedule = schedule
        self.model = model
        self.diffuse_time_t = torch.LongTensor([diffuse_time_t]).to(device)
        self.reference_flow_model = reference_flow_model

        self.lr = lr
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.use_wandb = use_wandb
        self.guassian_weight = gaussian_weight
        self.symmetry_punalty = symmetry_punalty

        self.dim = 1
        for s in shape:
            self.dim *= s
    
    def _check_flow_model(self, flow_model):
        # ensure there is no ReLU or other non-2rd-order-differentiable operations
        for layer in flow_model.modules():
            if isinstance(layer, nn.ReLU):
                raise ValueError("ReLU activations don't support 2nd-order differentiation. Use SiLU or other differentiable activations instead.")
    
    def estimate(self, x):
        xt = self.diffuser.add_noise(x, torch.randn_like(x), self.diffuse_time_t)
        xt = xt.reshape(xt.shape[0], -1)
        alpha = self.diffuser.add_noise(torch.ones(1), torch.zeros(1), self.diffuse_time_t).pow(2).item()
        return div_estimate(
                            self.flow_model, 
                            self.score_model, 
                            xt, 
                            num_samples=self.num_samples, 
                            )
    def global_loss(self, s, v):
        prod = torch.einsum('bn, bn -> b', s, v)
        return prod.mean().pow(2)
    
    def setup_wandb(self):
        if self.use_wandb:
            wandb.init(project='pattern-to-dynamics', 
                group=self.model,
                config={
                    'lr': self.lr,
                    'weight_decay': self.weight_decay,
                    'diffuser': dict(self.diffuser.config),
                    'num_samples': self.num_samples,
                }
            )
    
    def symmetry_loss(self, score, v):
        # l = score^\top \cdot v, score / v.shape = [batch, n]
        num_elements = score.numel()
        return torch.einsum('bn, bn -> b', score, v).sum() / num_elements
    
    @torch.no_grad()
    def compare_truth(self, x):
        """
        Compute the correlation between v and F(x), where
        F is the ground-truth of v. Both of the variables
        has the same shape of [batch, dim].

        corr = v @ F(x) / (|v| * |F(x)|)

        Here we are using the noise-free version to compute
        the correlation. This is important because randomness
        can lead to zero correlation at high-dimensional space.
        """
        x = x.reshape(x.shape[0], -1)
        v_f = self.flow_model(x)
        v_F = self.reference_flow_model(x)
        assert v_f.shape == v_F.shape
        assert len(v_f.shape) == 2
        return F.cosine_similarity(v_f, v_F, dim=1).mean().item()

    def train(self, epochs=10, batch_size=32):
        self.setup_wandb()
        optimizer = torch.optim.Adam(
            self.flow_model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        if self.schedule:
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=self.warmup_steps, 
                num_training_steps=epochs * len(dataloader))
        self.flow_model.train()

        lf = nn.MSELoss()
        step = 0
        
        for epoch in range(epochs):
            i = 0
            for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}/{epochs}"):
                step += 1
                batch = batch.to(self.device)
                if self.schedule:
                    self.scheduler.step()
                
                output = self.estimate(batch)
                v_avg_norm = output['v'].pow(2).sum(dim=-1).mean()
                loss_global = self.global_loss(output['score'], output['v']) / (v_avg_norm + 1e-8)
                div1, div2 = output['div(pv)_1'], output['div(pv)_2'] # [B]
                # loss_div = (div1 * div2 / self.dim).mean()
                # v.shape = [B, 2*H*W]
                loss_div = (div1 * div2).mean() / (v_avg_norm + 1e-8)
                loss_norm = (v_avg_norm / self.dim - 1).pow(2)
                loss_v = output['v'].mean().pow(2) / (v_avg_norm / self.dim)

                # reg = torch.mean(output['v@s'].pow(2) / (output['score'].norm(dim=1).pow(2) + 1e-6))
                loss = loss_div
                if self.symmetry_punalty > 0:
                    loss_symmetry = self.symmetry_loss(output['score'], output['v'])
                    loss += self.symmetry_punalty * loss_symmetry
                loss.backward()

                # Calculate correlation to the ground-truth
                if self.reference_flow_model is not None:
                    corr = self.compare_truth(batch)
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # average gradients
                    for param in self.flow_model.parameters():
                        if param.grad is not None:
                            param.grad /= self.gradient_accumulation_steps
                    # clip gradients
                    gradient_norm = torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    step = 0
                
                    if self.use_wandb:
                        info = {
                                'loss': loss_div.item(), 
                                'loss_global': loss_global.item(),
                                'epoch': epoch+i/len(dataloader), 
                                'lr': optimizer.param_groups[0]['lr'], 
                                'loss_symmetry': loss_symmetry.item() if self.symmetry_punalty > 0 else 0,
                                'v_avg': output['v'].mean().item(),
                                'v_norm': output['v'].pow(2).mean().sqrt().item(),
                                '|div(v)|': output['div(v)'].pow(2).mean().sqrt().item(),
                                '|v@s|': output['v@s'].pow(2).mean().sqrt().item(),
                                'gradient_norm': gradient_norm.item() if 'gradient_norm' in locals() else 0,
                            }
                        if self.reference_flow_model is not None:
                            info['corr'] = corr
                        wandb.log(info)
                i += 1
            
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
        
        if self.use_wandb:
            wandb.finish()