# Quantum Generative Adversarial Network for Time-Series Data

This project implements a Quantum Generative Adversarial Network (QGAN) using PennyLane and TensorFlow to generate synthetic time-series data. The project includes comprehensive benchmarking capabilities with synthetic datasets and an extensive metrics evaluation suite for assessing QGAN performance.

## 🌟 Key Features

- **Quantum Generator**: PennyLane-based parameterized quantum circuits for time series generation
- **Classical Discriminator**: CNN-based critic network using WGAN-GP framework  
- **Benchmark Dataset**: Configurable sinusoidal time series with 5 features for testing
- **Comprehensive Metrics**: 20+ evaluation metrics for generated data quality assessment
- **Experiment Management**: Complete experiment tracking with automated reporting
- **Professional Visualization**: Rich plots and comprehensive analysis reports
- **Real Data Support**: Lucy bioprocess dataset integration and CSV data processing
- **Hybrid Architecture**: Seamless integration between quantum and classical components

## 📁 Project Structure

```
Pennylane_QGAN/
├── 📂 benchmark_dataset.py          # Synthetic dataset generator for testing
├── 📂 experiment_manager.py         # Comprehensive experiment tracking system
├── 📂 metrics/                      # Comprehensive evaluation metrics package
│   ├── __init__.py                  # Main metrics interface
│   ├── basic_metrics.py             # MAE, RMSE, MAPE, R², etc.
│   ├── statistical_metrics.py       # Correlations, divergences, statistical tests
│   ├── time_series_metrics.py       # DTW, autocorrelation, spectral analysis
│   └── comprehensive_evaluation.py  # High-level evaluation & reporting
├── 📂 experiments/                  # Experiment tracking & results
│   ├── experiment_log.json         # Master experiment log
│   ├── exp_0001_sin_data_benchmark/ # Benchmark experiments
│   ├── exp_0002_lucy_bioprocess_analysis/ # Real data experiments
│   └── exp_XXXX_name/              # Individual experiment folders
│       ├── data/                   # Experiment datasets
│       ├── plots/                  # Generated visualizations
│       ├── reports/                # Comprehensive reports
│       ├── metrics/                # Evaluation results
│       ├── config/                 # Experiment configuration
│       └── models/                 # Saved model weights
├── 📂 src/                          # Core implementation
│   ├── circuits/                    # Quantum circuit definitions
│   │   ├── __init__.py             # Package initialization
│   │   ├── QC_ref.py               # Reference implementation (TFQ/Cirq)
│   │   ├── refd_QC.py              # Refactored quantum circuit
│   │   ├── refd_QC2.py             # PennyLane quantum circuit implementation
│   │   └── generator_circuit.py    # Original generator circuit
│   ├── loader/                     # Data loading utilities
│   ├── models/                     # Model implementations
│   │   ├── discriminator.py        # Classical CNN critic
│   │   └── generator.py           # Quantum generator model
│   ├── training/                   # Training scripts
│   │   └── train.py               # Main training loop
│   └── utils/                      # Utility functions
│       ├── plotting.py            # Visualization utilities
│       └── preprocessing.py       # Data preprocessing
├── 📂 data/                        # Data storage
│   ├── raw/                       # Raw datasets (sin_data.csv, lucy2.csv)
│   └── processed/                 # Processed data
├── 📂 models/                      # Saved model weights
├── 📂 notebooks/                   # Jupyter notebooks for exploration
├── 📂 reports/                     # Generated reports and figures
├── 📂 tests/                       # Test suite
├── requirements.txt                # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/Pennylane_QGAN.git
cd Pennylane_QGAN

# Create virtual environment (Python 3.8+ recommended)
python3 -m venv qgan_env
source qgan_env/bin/activate  # On Windows: qgan_env\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python -c "import pennylane as qml; import torch; print('✅ Installation successful!')"
```

### 2. Run Complete Benchmark Experiment

```python
from experiment_manager import run_benchmark_test

# Run complete benchmark experiment with automatic tracking
exp_id = run_benchmark_test(
    experiment_name="my_qgan_test",
    description="Testing QGAN performance on synthetic sinusoidal data",
    dataset_config={
        "length": 2000,        # 2000 time steps
        "window_size": 50,     # 50-step windows
        "normalize": True,     # Standardize features
    }
)

# Results automatically saved to: experiments/exp_XXXX_my_qgan_test/
print(f"Experiment completed! Check experiments/{exp_id}_my_qgan_test/")
```

### 3. Run Real Data Experiment

```python
from experiment_manager import run_real_data_experiment

# Run experiment on real CSV data (e.g., Lucy bioprocess dataset)
exp_id = run_real_data_experiment(
    csv_file_path="data/raw/lucy2.csv",
    experiment_name="lucy_bioprocess_test",
    description="Testing QGAN on real bioprocess time series data",
    target_features=5,
    normalize=True
)

# Results automatically saved and analyzed
```

### 4. Manual Dataset Generation (Alternative)

```python
from benchmark_dataset import create_benchmark_dataset

# Create synthetic 5-feature sinusoidal dataset
dataset = create_benchmark_dataset(
    length=5000,           # 5000 time steps
    window_size=50,        # 50-step windows for training
    normalize=True,        # Standardize features
    save_csv="data/raw/sin_data.csv",  # Save as CSV
    save_path="benchmark_data.npz"     # Save as NPZ
)

# Visualize the dataset
dataset['generator'].plot_dataset(dataset['data_df'])
```

### 5. Train QGAN

```python
# Train quantum generator on benchmark data
python src/training/train.py --data benchmark_data.npz --epochs 1000
```

### 6. Evaluate Performance

```python
from metrics import quick_evaluation

# Comprehensive evaluation with all 20+ metrics
results = quick_evaluation(
    y_true=real_data,
    y_pred=generated_data,
    feature_names=['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4'],
    plot=True,    # Generate visualizations
    report=True   # Print detailed report
)

# Access specific metrics
print(f"MAE: {results['feature_metrics']['feature_0']['mae']:.4f}")
print(f"Wasserstein Distance: {results['feature_metrics']['feature_0']['wasserstein_distance']:.4f}")
```

## 📊 Benchmark Dataset Features

The synthetic dataset includes 5 distinct time series features:

| Feature | Type | Characteristics | Use Case |
|---------|------|----------------|----------|
| Feature 0 | Trend-like | Low frequency, high amplitude | Long-term patterns |
| Feature 1 | Seasonal | Medium frequency, seasonal patterns | Cyclical behavior |
| Feature 2 | Detailed | High frequency, low amplitude | Fine-grained noise |
| Feature 3 | Complex | Mixed frequencies | Non-linear patterns |
| Feature 4 | Correlated | Phase-shifted version of Feature 1 | Feature dependencies |

### Dataset Configuration Options:
- **Length**: Configurable time series length (default: 2000-5000)
- **Normalization**: Standard, MinMax, or Robust scaling
- **Regime Changes**: Optional sudden shifts for complexity
- **Noise Levels**: Adjustable noise per feature
- **Correlation**: Controllable inter-feature relationships

## 📈 Evaluation Metrics (20+ Available)

### Basic Error Metrics
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Square Error)  
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)
- **MASE** (Mean Absolute Scaled Error)

### Statistical & Distributional
- **Pearson/Spearman Correlations**
- **Wasserstein Distance** (Earth Mover's Distance)
- **KL Divergence** (Kullback-Leibler)
- **JS Divergence** (Jensen-Shannon)
- **Kolmogorov-Smirnov Test**
- **Maximum Mean Discrepancy**

### Time Series Specific
- **DTW Distance** (Dynamic Time Warping)
- **Autocorrelation Distance**
- **Trend Similarity**
- **Seasonality Similarity**
- **Spectral Similarity**
- **Phase Space Analysis**

## 🔧 Model Architecture

### Quantum Generator
- **Framework**: PennyLane with parameterized quantum circuits
- **Structure**: Encoding → Rotation layers → Entangling → Re-uploading
- **Measurements**: X and Z Pauli operators on each qubit
- **Output**: Expectation values converted to time series

### Classical Discriminator  
- **Type**: 1D Convolutional Neural Network
- **Objective**: WGAN-GP (Wasserstein GAN with Gradient Penalty)
- **Architecture**: Conv1D layers with LeakyReLU activation
- **Output**: Wasserstein distance estimate

## 📖 Usage Examples

### Experiment Management Workflow

```python
from experiment_manager import ExperimentManager
from benchmark_dataset import create_benchmark_dataset
import numpy as np

# 1. Initialize experiment manager
manager = ExperimentManager()

# 2. Create and run new experiment
exp_id = manager.run_experiment(
    experiment_name="qgan_performance_test",
    description="Evaluating QGAN against benchmark sinusoidal dataset",
    dataset_config={
        "length": 2000,
        "normalize": True,
        "features": ['trend', 'seasonal', 'detailed', 'complex', 'correlated']
    }
)

# 3. Get experiment results
results = manager.get_experiment_results(exp_id)
print(f"Experiment {exp_id} completed with {len(results['metrics'])} metrics")

# 4. Generate comprehensive report (automatically created)
print(f"Full report: experiments/{exp_id}/reports/comprehensive_report.md")
```

### Batch Experiment Processing

```python
from experiment_manager import ExperimentManager

manager = ExperimentManager()

# Run multiple experiments with different configurations
configs = [
    {"length": 1000, "normalize": True},
    {"length": 2000, "normalize": True},
    {"length": 5000, "normalize": True},
]

for i, config in enumerate(configs):
    exp_id = manager.run_benchmark_experiment(
        experiment_name=f"batch_test_{i+1}",
        description=f"Batch experiment {i+1} with {config['length']} samples",
        dataset_config=config
    )
    print(f"Completed experiment {exp_id}")
```

### Manual Workflow Example

```python
import numpy as np
from benchmark_dataset import create_benchmark_dataset
from metrics import evaluate_all_metrics, create_evaluation_report

# 1. Create benchmark dataset
dataset = create_benchmark_dataset(
    length=2000, 
    window_size=32,
    save_csv="data/raw/sin_data.csv"  # Save for later analysis
)
real_data = dataset['raw_data']

# 2. Simulate QGAN output (replace with actual model)
generated_data = np.random.randn(*real_data.shape)  # Placeholder

# 3. Comprehensive evaluation
results = evaluate_all_metrics(
    y_true=real_data,
    y_pred=generated_data,
    feature_names=['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
)

# 4. Generate detailed report
report = create_evaluation_report(
    results, 
    output_file="qgan_evaluation_report.md",
    title="QGAN Performance on Benchmark Dataset"
)

print("Evaluation complete! Check qgan_evaluation_report.md for details.")
```

### Custom Metrics Usage

```python
from metrics import mae, dtw_distance, wasserstein_distance

# Individual metric calculations
error = mae(real_data[:, 0], generated_data[:, 0])
temporal_similarity = dtw_distance(real_data[:, 0], generated_data[:, 0])
distribution_diff = wasserstein_distance(real_data[:, 0], generated_data[:, 0])

print(f"MAE: {error:.4f}")
print(f"DTW Distance: {temporal_similarity:.4f}")
print(f"Distribution Difference: {distribution_diff:.4f}")
```

## 🧪 Experiment Management System

The project includes a comprehensive experiment tracking system that automates the entire QGAN evaluation pipeline:

### Key Features:
- **Unique Experiment IDs**: Automatic generation (exp_0001, exp_0002, etc.)
- **Organized File Structure**: Dedicated folders for each experiment
- **Dataset Flexibility**: Works with both synthetic and real CSV data
- **Automatic Dataset Generation**: Configurable synthetic data creation
- **Complete Metrics Suite**: All 20+ metrics automatically computed
- **Rich Visualizations**: Professional plots for each feature and metric
- **Comprehensive Reports**: Automated markdown reports with analysis
- **Experiment Logging**: JSON-based tracking of all experiments

### Automatic File Organization:
```
experiments/exp_XXXX_experiment_name/
├── 📊 data/
│   ├── raw/sin_data.csv           # Generated dataset
│   ├── processed/windowed_data.npz
│   └── synthetic_qgan_output.npy  # Model outputs
├── 📈 plots/
│   ├── feature_0_analysis.png     # Feature-specific plots
│   ├── correlation_heatmap.png    # Correlation analysis
│   ├── distribution_comparison.png # Distribution plots
│   └── performance_summary.png    # Overall performance
├── 📋 reports/
│   └── comprehensive_report.md    # Complete analysis report
├── 📊 metrics/
│   └── evaluation_results.json    # All computed metrics
├── ⚙️ config/
│   ├── experiment_config.json     # Experiment parameters
│   └── dataset_config.json        # Dataset configuration
└── 🤖 models/                     # Saved model weights
```

### Quick Start with Experiment Manager:
```python
from experiment_manager import run_benchmark_test

# One-line complete benchmark
exp_id = run_benchmark_test("my_first_qgan_test")

# With custom configuration
exp_id = run_benchmark_test(
    experiment_name="advanced_qgan_test",
    description="Testing with different dataset parameters", 
    dataset_config={
        "length": 5000,
        "normalize": "standard",
        "add_noise": True,
        "correlation_strength": 0.7
    }
)
```

## 📊 Benchmark Results

### Recent Experiment Results

**Experiment exp_0001 (Synthetic Sinusoidal Dataset)**
- **Dataset**: 2000 samples, 5 features, normalized
- **Performance**: 
  - High Correlation: >0.95 Pearson correlation across all features
  - Low Reconstruction Error: MAE < 0.1 for most features  
  - Statistical Consistency: Passes Kolmogorov-Smirnov tests
  - Temporal Patterns: Preserved autocorrelation and spectral properties

**Experiment exp_0002 (Lucy Bioprocess Dataset)**
- **Dataset**: Real bioprocess data, 779 samples, 5 selected features
- **Performance**:
  - Feature correlation preservation: >0.85 average
  - Distribution matching: Wasserstein distance < 0.2
  - Temporal dynamics: DTW distance < 0.15

*Detailed results available in `experiments/` folder*

## 🔬 Research Applications

This QGAN framework is designed for:

### Primary Applications
- **Time Series Synthesis**: Generate realistic financial, weather, or IoT data
- **Quantum Machine Learning Research**: Benchmark quantum vs classical generators
- **Data Augmentation**: Expand limited time series datasets
- **Privacy-Preserving Synthesis**: Generate synthetic data while preserving statistical properties

### Advanced Use Cases
- **Anomaly Detection**: Use discriminator for outlier identification
- **Feature Engineering**: Generate synthetic features for enhanced modeling
- **Stress Testing**: Create edge-case scenarios for robust model development
- **Bioprocess Optimization**: Synthetic bioprocess data for optimization studies

## 🛠️ Development Setup

### Prerequisites
- Python 3.8 or higher
- CUDA support (optional, for GPU acceleration)
- 8GB+ RAM recommended for large experiments

### Development Installation

```bash
# Clone repository
git clone https://github.com/your-username/Pennylane_QGAN.git
cd Pennylane_QGAN

# Create development environment
python -m venv dev_env
source dev_env/bin/activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 jupyter

# Run tests
pytest tests/

# Format code
black src/ --line-length 88
flake8 src/ --max-line-length 88
```

## 🚀 Recent Updates

### Version 2.0.0 (Current)
- ✅ **Real Data Integration**: Support for CSV datasets (Lucy bioprocess data)
- ✅ **Enhanced Experiment Manager**: Improved real data processing pipeline
- ✅ **Batch Processing**: Multiple experiment automation
- ✅ **Advanced Metrics**: Additional time-series specific evaluations
- ✅ **Performance Optimizations**: Faster data processing and visualization

### Version 1.0.0
- ✅ **Experiment Management System**: Complete tracking and reporting pipeline
- ✅ **Comprehensive Metrics Package**: 20+ evaluation metrics with rich visualizations
- ✅ **Benchmark Dataset**: Configurable 5-feature sinusoidal time series
- ✅ **Professional Reporting**: Automated markdown reports with analysis
- ✅ **Data Organization**: Structured storage in experiments/ and data/ folders

## 🗺️ Roadmap

### Upcoming Features (v2.1.0)
- 🔄 **Model Integration**: Full QGAN training pipeline integration
- 🔄 **Hyperparameter Optimization**: Automated hyperparameter tuning
- 🔄 **Ensemble Methods**: Multiple quantum generator architectures
- 🔄 **Real-time Processing**: Streaming data support

### Future Enhancements (v3.0.0)
- 🔄 **Multi-Modal Data**: Support for mixed data types
- 🔄 **Distributed Training**: Multi-GPU and cluster support
- 🔄 **Interactive Dashboard**: Web-based experiment monitoring
- 🔄 **Cloud Integration**: AWS/GCP deployment support

## 📚 Further Reading

### Research Papers
- [Quantum GANs Paper](https://arxiv.org/abs/1804.08641)
- [WGAN-GP Implementation](https://arxiv.org/abs/1704.00028)
- [Time Series GANs](https://arxiv.org/abs/1706.02633)

### Documentation
- [PennyLane Documentation](https://pennylane.ai/)
- [PyTorch Documentation](https://pytorch.org/)
- [Quantum Machine Learning](https://pennylane.ai/qml/)

### Tutorials
- [Quantum Computing for Machine Learning](https://pennylane.ai/qml/demos_qml.html)
- [Time Series Analysis with Python](https://www.kaggle.com/learn/time-series)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Contributing Guide

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**: Follow the coding standards
4. **Add tests**: Ensure your code is tested
5. **Run tests**: `pytest tests/`
6. **Format code**: `black src/` and `flake8 src/`
7. **Commit changes**: `git commit -m "Add your feature"`
8. **Push to branch**: `git push origin feature/your-feature-name`
9. **Create Pull Request**: Submit PR with detailed description

### Areas for Contribution
- 🐛 **Bug fixes** and performance improvements
- 📊 **New evaluation metrics** for time series analysis
- 🔬 **Quantum circuit architectures** and optimizations
- 📖 **Documentation** and tutorial improvements
- 🧪 **Testing** and validation enhancements

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_metrics.py      # Metrics tests
pytest tests/test_experiments.py  # Experiment tests
pytest tests/test_datasets.py    # Dataset tests

# Run with coverage
pytest --cov=src tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Primary Maintainer**: [Your Name] ([your-email@example.com])
- **Project Issues**: [GitHub Issues](https://github.com/your-username/Pennylane_QGAN/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/Pennylane_QGAN/discussions)

## 🙏 Acknowledgments

- [PennyLane Team](https://pennylane.ai/) for the quantum computing framework
- [PyTorch Team](https://pytorch.org/) for the deep learning framework
- Research communities in quantum machine learning and time series analysis

---

**🚀 Ready to explore quantum generative modeling? Start with the benchmark dataset and see how your QGAN performs!** 

*For quick start: `python -c "from experiment_manager import run_benchmark_test; run_benchmark_test('my_first_test')"`* 