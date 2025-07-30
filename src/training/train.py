import os
import torch
from torch import optim
from torch.utils.data import DataLoader
import pennylane as qml
import matplotlib.pyplot as plt
from typing import Dict, Optional, Any

from src.loader.dataloader import load_processed_data
from src.models.generator import QuantumGenerator
from src.models.refd_generator2 import QuantumGenerator as HQuantumGenerator
from src.models.discriminator import Discriminator
from src.utils.preprocessing import TimeSeriesDataset

def gradient_penalty(critic, real_samples, fake_samples):
    """Calculate gradient penalty for WGAN-GP"""
    batch_size, seq_len = real_samples.shape
    alpha = torch.rand(batch_size, 1).expand(batch_size, seq_len)
    
    interpolated = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    c_interpolated = critic(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=c_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(c_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp

def get_generator_model(generator_type: str, n_qubits: int, n_layers: int, window_size: int, q_device: Any):
    """
    Factory function to create the appropriate generator model based on type.
    
    Args:
        generator_type: Type of generator ('original' or 'h_generator')
        n_qubits: Number of qubits
        n_layers: Number of layers
        window_size: Size of the output window
        q_device: PennyLane device
        
    Returns:
        Generator model instance
    """
    if generator_type == 'original':
        return QuantumGenerator(n_qubits, n_layers, window_size, q_device)
    elif generator_type == 'h_generator':
        return HQuantumGenerator(n_qubits, n_layers, window_size, q_device)
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")

def train_qgan(config: Optional[Dict] = None):
    """
    Train the QGAN with configurable parameters and generator architecture.
    
    Args:
        config: Dictionary containing training configuration. If None, uses default values.
    """
    # Default configuration
    default_config = {
        'n_qubits': 5,
        'n_layers': 6,
        'window_size': 10,
        'device_name': "default.qubit",
        'generator_type': 'original',  # 'original' or 'h_generator'
        'lr': 0.0001,
        'betas': (0.5, 0.999),
        'n_epochs': 100,
        'batch_size': 32,
        'n_critic': 5,
        'lambda_gp': 10,
        'processed_data_path': 'data/processed/log_returns.pt'
    }

    # Update config with provided values
    if config:
        default_config.update(config)
    config = default_config

    # Setup directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports/figures', exist_ok=True)

    # Initialize device
    q_device = qml.device(config['device_name'], wires=config['n_qubits'])

    # Data
    timeseries_data = load_processed_data(config['processed_data_path']).numpy()
    dataset = TimeSeriesDataset(timeseries_data, config['window_size'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)

    # Models
    generator = get_generator_model(
        config['generator_type'],
        config['n_qubits'],
        config['n_layers'],
        config['window_size'],
        q_device
    )
    critic = Discriminator(window_size=config['window_size'])

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=config['lr'], betas=config['betas'])
    d_optimizer = optim.Adam(critic.parameters(), lr=config['lr'], betas=config['betas'])
    
    print(f"Training with {config['generator_type']} generator architecture")
    print("Generator parameters:", sum(p.numel() for p in generator.parameters()))
    print("Critic parameters:", sum(p.numel() for p in critic.parameters()))
    print("\nStarting training...")

    # Training loop
    g_losses = []
    d_losses = []
    for epoch in range(config['n_epochs']):
        for i, real_samples in enumerate(dataloader):
            # Train Critic
            for _ in range(config['n_critic']):
                d_optimizer.zero_grad()
                noise = torch.randn(config['batch_size'], config['n_qubits'])
                fake_samples = generator(noise)
                
                c_real = critic(real_samples).reshape(-1)
                c_fake = critic(fake_samples.detach()).reshape(-1)
                
                loss_c = -torch.mean(c_real) + torch.mean(c_fake)
                gp = gradient_penalty(critic, real_samples, fake_samples.detach())
                d_loss = loss_c + config['lambda_gp'] * gp
                
                d_loss.backward()
                d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()
            noise = torch.randn(config['batch_size'], config['n_qubits'])
            fake_samples = generator(noise)
            g_loss = -torch.mean(critic(fake_samples))
            g_loss.backward()
            g_optimizer.step()
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
    
        print(f"Epoch [{epoch+1}/{config['n_epochs']}] | Critic Loss: {d_losses[-1]:.4f} | Generator Loss: {g_losses[-1]:.4f}")

    # Save model and plot losses
    model_path = f"models/qgan_generator_{config['generator_type']}.pth"
    torch.save(generator.state_dict(), model_path)
    print(f"\nTraining complete. Generator model saved to '{model_path}'")

    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Critic Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(f'Training Losses ({config["generator_type"]} Generator)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'reports/figures/training_losses_{config["generator_type"]}.png')
    print(f"Training loss plot saved to 'reports/figures/training_losses_{config['generator_type']}.png'")

if __name__ == "__main__":
    # Example usage with H-generator
    config = {
        'generator_type': 'h_generator',
        'n_epochs': 100,
        'n_qubits': 5,
        'n_layers': 6
    }
    train_qgan(config) 