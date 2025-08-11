#!/usr/bin/env python3
"""
QGAN Generation Pipeline Debugger
=================================

This script provides comprehensive debugging for the QGAN generation pipeline
to identify and fix issues causing blank plots and invalid synthetic data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

def debug_generation_pipeline(qgan, OD_log_delta, transformed_norm_OD_log_delta, 
                            WINDOW_LENGTH, NUM_QUBITS, verbose=True):
    """
    Comprehensive debugging of the QGAN generation pipeline
    
    Args:
        qgan: Trained QGAN model
        OD_log_delta: Original log returns tensor
        transformed_norm_OD_log_delta: Preprocessed data tensor
        WINDOW_LENGTH: Window size for generation
        NUM_QUBITS: Number of qubits in quantum circuit
        verbose: Print detailed debug information
    
    Returns:
        dict: Debug results and generated data
    """
    debug_results = {}
    
    if verbose:
        print("=" * 60)
        print("🔍 QGAN GENERATION PIPELINE DEBUGGER")
        print("=" * 60)
    
    # Step 1: Verify model training status
    if verbose:
        print("\n1️⃣ TRAINING STATUS CHECK")
        print("-" * 30)
    
    epochs_completed = len(qgan.critic_loss_avg)
    param_range = [qgan.params_pqc.min().item(), qgan.params_pqc.max().item()]
    
    debug_results['training_status'] = {
        'epochs_completed': epochs_completed,
        'parameter_range': param_range,
        'is_trained': epochs_completed > 0
    }
    
    if verbose:
        print(f"✅ Epochs completed: {epochs_completed}")
        print(f"✅ Parameter range: [{param_range[0]:.4f}, {param_range[1]:.4f}]")
        
        if epochs_completed == 0:
            print("❌ MODEL NOT TRAINED!")
            return debug_results
        else:
            print("✅ Model appears trained")
    
    # Step 2: Check input data validity
    if verbose:
        print("\n2️⃣ INPUT DATA VALIDATION")
        print("-" * 30)
    
    try:
        od_stats = {
            'shape': OD_log_delta.shape,
            'range': [OD_log_delta.min().item(), OD_log_delta.max().item()],
            'mean': OD_log_delta.mean().item(),
            'std': OD_log_delta.std().item(),
            'has_nan': torch.isnan(OD_log_delta).any().item(),
            'has_inf': torch.isinf(OD_log_delta).any().item()
        }
        
        transformed_stats = {
            'shape': transformed_norm_OD_log_delta.shape,
            'range': [transformed_norm_OD_log_delta.min().item(), transformed_norm_OD_log_delta.max().item()],
            'mean': transformed_norm_OD_log_delta.mean().item(),
            'std': transformed_norm_OD_log_delta.std().item(),
            'has_nan': torch.isnan(transformed_norm_OD_log_delta).any().item(),
            'has_inf': torch.isinf(transformed_norm_OD_log_delta).any().item()
        }
        
        debug_results['input_data'] = {
            'original': od_stats,
            'transformed': transformed_stats
        }
        
        if verbose:
            print(f"✅ Original data: {od_stats['shape']}, range [{od_stats['range'][0]:.6f}, {od_stats['range'][1]:.6f}]")
            print(f"✅ Transformed data: {transformed_stats['shape']}, range [{transformed_stats['range'][0]:.6f}, {transformed_stats['range'][1]:.6f}]")
            
            if od_stats['has_nan'] or od_stats['has_inf']:
                print("❌ Original data contains NaN/Inf values!")
            if transformed_stats['has_nan'] or transformed_stats['has_inf']:
                print("❌ Transformed data contains NaN/Inf values!")
                
    except Exception as e:
        if verbose:
            print(f"❌ Error checking input data: {e}")
        debug_results['input_data'] = {'error': str(e)}
        return debug_results
    
    # Step 3: Test quantum generator directly
    if verbose:
        print("\n3️⃣ QUANTUM GENERATOR TEST")
        print("-" * 30)
    
    try:
        # Test single generation
        test_noise = np.random.uniform(0, 2 * np.pi, size=NUM_QUBITS)
        test_input = torch.tensor(test_noise, dtype=torch.float32)
        
        with torch.no_grad():  # Disable gradients for testing
            test_output = qgan.generator(test_input, qgan.params_pqc)
            
        if isinstance(test_output, list):
            test_output = torch.stack(test_output)
            
        generator_stats = {
            'output_shape': test_output.shape,
            'output_range': [test_output.min().item(), test_output.max().item()],
            'output_mean': test_output.mean().item(),
            'output_std': test_output.std().item(),
            'has_nan': torch.isnan(test_output).any().item(),
            'has_inf': torch.isinf(test_output).any().item(),
            'sample_values': test_output[:5].detach().cpu().numpy().tolist()
        }
        
        debug_results['generator_test'] = generator_stats
        
        if verbose:
            print(f"✅ Generator output shape: {generator_stats['output_shape']}")
            print(f"✅ Generator output range: [{generator_stats['output_range'][0]:.6f}, {generator_stats['output_range'][1]:.6f}]")
            print(f"✅ Sample values: {generator_stats['sample_values']}")
            
            if generator_stats['has_nan'] or generator_stats['has_inf']:
                print("❌ Generator output contains NaN/Inf!")
            if generator_stats['output_range'][0] == generator_stats['output_range'][1]:
                print("❌ Generator output is constant (all same values)!")
                
    except Exception as e:
        if verbose:
            print(f"❌ Error testing generator: {e}")
        debug_results['generator_test'] = {'error': str(e)}
        return debug_results
    
    # Step 4: Full generation pipeline test
    if verbose:
        print("\n4️⃣ FULL GENERATION PIPELINE")
        print("-" * 30)
    
    try:
        # Generate samples
        num_samples = len(OD_log_delta) // WINDOW_LENGTH
        if verbose:
            print(f"Generating {num_samples} samples with window length {WINDOW_LENGTH}")
        
        input_circuits_batch = []
        for _ in range(min(num_samples, 10)):  # Limit to 10 for debugging
            noise_values = np.random.uniform(0, 2 * np.pi, size=NUM_QUBITS)
            input_circuits_batch.append(noise_values)
        
        generator_inputs = torch.stack([torch.tensor(noise, dtype=torch.float32) for noise in input_circuits_batch])
        
        # Generate batch
        batch_generated = []
        for i, generator_input in enumerate(generator_inputs):
            with torch.no_grad():
                generated_sample = qgan.generator(generator_input, qgan.params_pqc)
                if isinstance(generated_sample, list):
                    generated_sample = torch.stack(generated_sample)
                batch_generated.append(generated_sample.to(torch.float64))
        
        batch_generated = torch.stack(batch_generated)
        
        # Reshape and process
        generated_data = torch.reshape(batch_generated, shape=(len(batch_generated) * WINDOW_LENGTH,))
        
        pipeline_stats = {
            'num_samples_generated': len(batch_generated),
            'batch_shape': batch_generated.shape,
            'reshaped_shape': generated_data.shape,
            'generated_range': [generated_data.min().item(), generated_data.max().item()],
            'generated_mean': generated_data.mean().item(),
            'generated_std': generated_data.std().item()
        }
        
        debug_results['pipeline_test'] = pipeline_stats
        
        if verbose:
            print(f"✅ Generated {pipeline_stats['num_samples_generated']} samples")
            print(f"✅ Batch shape: {pipeline_stats['batch_shape']}")
            print(f"✅ Final shape: {pipeline_stats['reshaped_shape']}")
            print(f"✅ Generated range: [{pipeline_stats['generated_range'][0]:.6f}, {pipeline_stats['generated_range'][1]:.6f}]")
        
        # Test preprocessing reversal
        if verbose:
            print("\n5️⃣ PREPROCESSING REVERSAL TEST")
            print("-" * 30)
        
        # Import necessary functions (assuming they're available)
        try:
            from qgan_pennylane import rescale, lambert_w_transform, denormalize
            
            # Rescale
            rescaled_data = rescale(generated_data, transformed_norm_OD_log_delta)
            rescale_stats = {
                'range': [rescaled_data.min().item(), rescaled_data.max().item()],
                'mean': rescaled_data.mean().item()
            }
            
            # Lambert W transform
            lambert_data = lambert_w_transform(rescaled_data, 1)
            lambert_stats = {
                'range': [lambert_data.min().item(), lambert_data.max().item()],
                'mean': lambert_data.mean().item()
            }
            
            # Denormalize
            final_data = denormalize(lambert_data, torch.mean(OD_log_delta), torch.std(OD_log_delta))
            final_stats = {
                'range': [final_data.min().item(), final_data.max().item()],
                'mean': final_data.mean().item(),
                'std': final_data.std().item()
            }
            
            debug_results['preprocessing_reversal'] = {
                'rescale': rescale_stats,
                'lambert': lambert_stats,
                'final': final_stats
            }
            
            if verbose:
                print(f"✅ After rescale: [{rescale_stats['range'][0]:.6f}, {rescale_stats['range'][1]:.6f}]")
                print(f"✅ After Lambert W: [{lambert_stats['range'][0]:.6f}, {lambert_stats['range'][1]:.6f}]")
                print(f"✅ Final result: [{final_stats['range'][0]:.6f}, {final_stats['range'][1]:.6f}]")
            
            # Store final data for plotting
            debug_results['final_generated_data'] = final_data
            
        except ImportError as e:
            if verbose:
                print(f"❌ Cannot import preprocessing functions: {e}")
            debug_results['preprocessing_reversal'] = {'error': f'Import error: {e}'}
        except Exception as e:
            if verbose:
                print(f"❌ Error in preprocessing reversal: {e}")
            debug_results['preprocessing_reversal'] = {'error': str(e)}
    
    except Exception as e:
        if verbose:
            print(f"❌ Error in full pipeline: {e}")
        debug_results['pipeline_test'] = {'error': str(e)}
    
    if verbose:
        print("\n" + "=" * 60)
        print("🎯 DEBUGGING COMPLETE")
        print("=" * 60)
    
    return debug_results

def create_diagnostic_plots(debug_results, OD_log_delta, save_plots=True):
    """
    Create diagnostic plots based on debug results
    """
    if 'final_generated_data' not in debug_results:
        print("❌ No final generated data available for plotting")
        return
    
    fake_data = debug_results['final_generated_data']
    real_data = OD_log_delta
    
    # Convert to numpy
    real_np = real_data.detach().cpu().numpy() if isinstance(real_data, torch.Tensor) else np.array(real_data)
    fake_np = fake_data.detach().cpu().numpy() if isinstance(fake_data, torch.Tensor) else np.array(fake_data)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Time series comparison
    axes[0, 0].plot(real_np[:min(len(real_np), 500)], 'b-', label='Real', alpha=0.7)
    axes[0, 0].plot(fake_np[:min(len(fake_np), 500)], 'r-', label='Generated', alpha=0.7)
    axes[0, 0].set_title('Time Series Comparison (First 500 points)')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Log Returns')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot 2: Histograms
    axes[0, 1].hist(real_np, bins=50, alpha=0.7, label='Real', density=True)
    axes[0, 1].hist(fake_np, bins=50, alpha=0.7, label='Generated', density=True)
    axes[0, 1].set_title('Distribution Comparison')
    axes[0, 1].set_xlabel('Log Returns')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot 3: Q-Q plot
    from scipy import stats
    stats.probplot(fake_np, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot of Generated Data')
    axes[1, 0].grid(True)
    
    # Plot 4: Statistics comparison
    real_stats = [np.mean(real_np), np.std(real_np), np.min(real_np), np.max(real_np)]
    fake_stats = [np.mean(fake_np), np.std(fake_np), np.min(fake_np), np.max(fake_np)]
    
    x = np.arange(4)
    width = 0.35
    
    axes[1, 1].bar(x - width/2, real_stats, width, label='Real', alpha=0.7)
    axes[1, 1].bar(x + width/2, fake_stats, width, label='Generated', alpha=0.7)
    axes[1, 1].set_title('Statistics Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Mean', 'Std', 'Min', 'Max'])
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('qgan_diagnostic_plots.png', dpi=300, bbox_inches='tight')
        print("✅ Diagnostic plots saved as 'qgan_diagnostic_plots.png'")
    
    plt.show()
    
    # Print summary statistics
    print("\n📊 SUMMARY STATISTICS")
    print("-" * 40)
    print(f"Real data    - Mean: {np.mean(real_np):.6f}, Std: {np.std(real_np):.6f}")
    print(f"Generated    - Mean: {np.mean(fake_np):.6f}, Std: {np.std(fake_np):.6f}")
    print(f"Real range   - [{np.min(real_np):.6f}, {np.max(real_np):.6f}]")
    print(f"Generated    - [{np.min(fake_np):.6f}, {np.max(fake_np):.6f}]")

def fix_generation_pipeline():
    """
    Provides step-by-step instructions to fix common generation issues
    """
    print("\n🔧 GENERATION PIPELINE FIX GUIDE")
    print("=" * 50)
    
    fixes = [
        "1. Ensure model is trained before generation",
        "2. Check for NaN/Inf values in quantum generator output",
        "3. Verify preprocessing functions are imported correctly",
        "4. Test each step of the pipeline individually",
        "5. Use torch.no_grad() during generation to avoid memory issues",
        "6. Check that rescaling uses correct reference data",
        "7. Verify Lambert W transform parameters",
        "8. Ensure denormalization uses correct mean/std from original data",
        "9. Check plot axis limits and data ranges",
        "10. Add explicit error handling for each step"
    ]
    
    for fix in fixes:
        print(f"✅ {fix}")
    
    print("\n💡 QUICK FIXES:")
    print("- Add debug prints after each processing step")
    print("- Use smaller batch sizes for testing")
    print("- Check variable scope in notebook cells")
    print("- Verify all required functions are imported")
    print("- Test with dummy data first")

if __name__ == "__main__":
    print("🔍 QGAN Generation Pipeline Debugger")
    print("Import this module and use debug_generation_pipeline() function")
    print("Example usage:")
    print("  from debug_generation import debug_generation_pipeline, create_diagnostic_plots")
    print("  results = debug_generation_pipeline(qgan, OD_log_delta, transformed_norm_OD_log_delta, WINDOW_LENGTH, NUM_QUBITS)")
    print("  create_diagnostic_plots(results, OD_log_delta)")
