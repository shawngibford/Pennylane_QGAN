#!/usr/bin/env python3
"""
Test script to verify the generation pipeline fixes
Run this after training your QGAN model to test the generation pipeline
"""

def test_generation_pipeline():
    """
    Test the fixed generation pipeline
    """
    print("🧪 TESTING GENERATION PIPELINE FIXES")
    print("=" * 50)
    
    # This script assumes you have already:
    # 1. Loaded your data
    # 2. Trained your QGAN model
    # 3. Have the following variables available:
    #    - qgan (trained model)
    #    - OD_log_delta (original data)
    #    - transformed_norm_OD_log_delta (preprocessed data)
    #    - WINDOW_LENGTH, NUM_QUBITS
    
    try:
        # Import required modules
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        
        print("✅ All required modules imported successfully")
        
        # Check if model is trained
        if len(qgan.critic_loss_avg) == 0:
            print("❌ ERROR: Model not trained! Run training first.")
            return False
        
        print(f"✅ Model trained for {len(qgan.critic_loss_avg)} epochs")
        
        # Test generation with a small batch
        print("\n🔄 Testing generation with small batch...")
        
        # Generate 5 samples for testing
        test_samples = 5
        input_circuits_batch = []
        
        for _ in range(test_samples):
            noise_values = np.random.uniform(0, 2 * np.pi, size=NUM_QUBITS)
            input_circuits_batch.append(noise_values)
        
        generator_inputs = torch.stack([torch.tensor(noise, dtype=torch.float32) for noise in input_circuits_batch])
        
        # Generate samples
        batch_generated = []
        for i, generator_input in enumerate(generator_inputs):
            with torch.no_grad():
                generated_sample = qgan.generator(generator_input, qgan.params_pqc)
                if isinstance(generated_sample, list):
                    generated_sample = torch.stack(generated_sample)
                batch_generated.append(generated_sample.to(torch.float64))
                
                # Check each sample
                sample_min, sample_max = generated_sample.min().item(), generated_sample.max().item()
                print(f"  Sample {i}: range [{sample_min:.6f}, {sample_max:.6f}]")
                
                if torch.isnan(generated_sample).any():
                    print(f"  ❌ Sample {i} contains NaN!")
                    return False
                if sample_min == sample_max:
                    print(f"  ❌ Sample {i} is constant!")
                    return False
        
        batch_generated = torch.stack(batch_generated)
        print(f"✅ Generated batch shape: {batch_generated.shape}")
        
        # Test preprocessing pipeline
        print("\n🔄 Testing preprocessing pipeline...")
        
        # Reshape
        generated_data = torch.reshape(batch_generated, shape=(test_samples * WINDOW_LENGTH,))
        print(f"✅ Reshaped to: {generated_data.shape}")
        
        # Rescale
        generated_data_rescaled = rescale(generated_data, transformed_norm_OD_log_delta)
        print(f"✅ Rescaled range: [{generated_data_rescaled.min().item():.6f}, {generated_data_rescaled.max().item():.6f}]")
        
        # Lambert W transform
        original_norm = lambert_w_transform(generated_data_rescaled, 1)
        print(f"✅ Lambert W range: [{original_norm.min().item():.6f}, {original_norm.max().item():.6f}]")
        
        # Denormalize
        fake_original = denormalize(original_norm, torch.mean(OD_log_delta), torch.std(OD_log_delta))
        print(f"✅ Final range: [{fake_original.min().item():.6f}, {fake_original.max().item():.6f}]")
        
        # Final validation
        if torch.isnan(fake_original).any() or torch.isinf(fake_original).any():
            print("❌ Final data contains NaN/Inf!")
            return False
        
        if fake_original.min().item() == fake_original.max().item():
            print("❌ Final data is constant!")
            return False
        
        print("✅ All preprocessing steps completed successfully!")
        
        # Test plotting
        print("\n🔄 Testing plotting...")
        
        # Convert to numpy
        OD_log_delta_np = OD_log_delta.detach().cpu().numpy()
        fake_OD_log_delta_np = fake_original.detach().cpu().numpy()
        
        # Create test plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot original
        axes[0].plot(OD_log_delta_np[:100], 'b-', alpha=0.8)  # First 100 points
        axes[0].set_title('Original (First 100 points)')
        axes[0].grid(True)
        
        # Plot generated
        axes[1].plot(fake_OD_log_delta_np, 'r-', alpha=0.8)
        axes[1].set_title('Generated (Test Batch)')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Test plots created successfully!")
        
        # Summary statistics
        print(f"\n📊 SUMMARY:")
        print(f"Original data  - Mean: {np.mean(OD_log_delta_np):.6f}, Std: {np.std(OD_log_delta_np):.6f}")
        print(f"Generated data - Mean: {np.mean(fake_OD_log_delta_np):.6f}, Std: {np.std(fake_OD_log_delta_np):.6f}")
        
        print(f"\n🎉 SUCCESS! Generation pipeline is working correctly!")
        print("You can now run the full generation in your main script.")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Generation Pipeline Test Script")
    print("Make sure you have trained your QGAN model first!")
    print("Then run: test_generation_pipeline()")
