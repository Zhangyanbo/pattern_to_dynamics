def integrate_rk4(vf, pos, dt):
    """
    Performs one step of the 4th-order Runge-Kutta integration method.
    
    Args:
        vf: A function that takes a position tensor and returns a velocity vector.
        pos: The current position tensor.
        dt: The time step for integration (dt).
        
    Returns:
        The new position after one integration step.
    """
    k1 = vf(pos)
    k2 = vf(pos + k1 * dt / 2)
    k3 = vf(pos + k2 * dt / 2)
    k4 = vf(pos + k3 * dt)
    return pos + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6


def linearized_flow(flow_model, input_shape, **kwargs):
    """
    Convert the flow model from f: [B, C, H, W] -> [B, C, H, W] to f: [B, C*H*W] -> [B, C*H*W]
    """
    def linearized(x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, *input_shape)
        return flow_model(x, **kwargs).reshape(batch_size, -1)
    return linearized

def linearized_diffusion(diffusion_model, input_shape, **kwargs):
    """
    Convert the diffusion model from f: [B, C, H, W] -> [B, C, H, W] to f: [B, C*H*W] -> [B, C*H*W]
    """
    def linearized(x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, *input_shape)
        return diffusion_model(x, **kwargs).sample.reshape(batch_size, -1)
    return linearized

def vector_linearize(v):
    """
    Convert the vector from [B, C, H, W] to [B, C*H*W]
    """
    batch_size = v.shape[0]
    return v.reshape(batch_size, -1)

class Logger:
    """
    A logger for tracking and averaging values over time.
    
    This class maintains a buffer of values and computes their average over time.

    Example Usage:
    ```python
    logger = Logger()
    for epoch in range(10):
        for i in range(1000):
            logger.log('loss', i)
        logger.step()
    print(logger['loss'])
    ```
    """

    def __init__(self):
        self.buffer = {}
        self.history = {}
    
    def log(self, name, value):
        if name not in self.buffer:
            self.buffer[name] = []
        self.buffer[name].append(value)
    
    def step(self):
        for name, values in self.buffer.items():
            if name not in self.history:
                self.history[name] = []
            if values:
                avg_value = sum(values) / len(values)
                self.history[name].append(avg_value)
        self.buffer.clear()
    
    def __getitem__(self, key):
        return self.history.get(key, [])