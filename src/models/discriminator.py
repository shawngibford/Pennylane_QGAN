from torch import nn

class Discriminator(nn.Module):
    """
    Classical critic model for WGAN-GP.
    Note: No final Sigmoid layer, as it outputs a score, not a probability.
    """
    def __init__(self, window_size):
        """
        Args:
            window_size (int): The size of the input window.
        """
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(window_size, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            
            nn.Linear(16, 1)
            # No Sigmoid for WGAN-GP
        )
    
    def forward(self, x):
        return self.model(x) 