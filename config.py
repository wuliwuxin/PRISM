"""
PRISM Configuration Manager
Flexible configuration for different prediction modes and experiments
"""
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    # Random seeds for reproducibility
    seeds: List[int] = field(default_factory=lambda: [42, 2024, 123456, 2025, 2026])

    # Prediction horizons (in hours)
    pred_lens: List[int] = field(default_factory=lambda: [6, 12, 24, 48])

    # GPU devices
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1])

    # Prediction mode: 'total', 'priority', 'organization'
    prediction_mode: str = 'total'

    # Ablation study flag
    run_ablation: bool = False  # Default: do not run ablation

    # Data configuration
    seq_len: int = 96  # Input sequence length (4 days)
    time_window: int = 3600  # Time window in seconds (1 hour)

    # Model hyperparameters
    batch_size: int = 128
    epochs: int = 100
    patience: int = 20

    use_patch: bool = True
    patch_len: int = 16
    stride: int = 8

    d_model: int = 256
    n_heads: int = 8
    e_layers: int = 3
    d_ff: int = 1024
    n_primitives: int = 16

    dropout: float = 0.1
    lr: float = 0.001
    lambda_1: float = 0.1  # MAE loss weight
    lambda_div: float = 0.01  # Diversity loss weight

    # Data split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Paths
    data_dir: str = 'data'
    checkpoint_dir: str = 'checkpoints'
    results_dir: str = 'results'
    visualization_dir: str = 'visualizations'
    predictions_dir: str = 'predictions'  # For saving .npy files
    log_dir: str = 'logs'

    def __post_init__(self):
        """Validate configuration"""
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "Data split ratios must sum to 1.0"
        assert self.prediction_mode in ['total', 'priority', 'organization'], \
            "prediction_mode must be 'total', 'priority', or 'organization'"
        assert len(self.gpu_ids) > 0, "At least one GPU must be specified"

        # Check GPU availability
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            for gpu_id in self.gpu_ids:
                if gpu_id >= available_gpus:
                    print(f"Warning: GPU {gpu_id} not available. Available GPUs: {available_gpus}")
        else:
            print("Warning: CUDA not available. Will use CPU.")

    def get_device(self, gpu_id: int) -> torch.device:
        """Get device for computation"""
        if torch.cuda.is_available() and gpu_id in self.gpu_ids:
            return torch.device(f'cuda:{gpu_id}')
        return torch.device('cpu')

    def get_model_config(self) -> Dict:
        """Get model configuration as dictionary"""
        return {
            'seq_len': self.seq_len,
            'use_patch': self.use_patch,
            'patch_len': self.patch_len,
            'stride': self.stride,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'e_layers': self.e_layers,
            'd_ff': self.d_ff,
            'n_primitives': self.n_primitives,
            'dropout': self.dropout
        }

    def get_training_config(self) -> Dict:
        """Get training configuration as dictionary"""
        return {
            'epochs': self.epochs,
            'lr': self.lr,
            'patience': self.patience,
            'lambda_1': self.lambda_1,
            'lambda_div': self.lambda_div
        }

    def print_config(self):
        """Print configuration summary"""
        print("\n" + "=" * 80)
        print("EXPERIMENT CONFIGURATION")
        print("=" * 80)

        print("\nExperiment Settings:")
        print(f"  Seeds: {self.seeds}")
        print(f"  Prediction Lengths: {self.pred_lens} hours")
        print(f"  GPU IDs: {self.gpu_ids}")
        print(f"  Prediction Mode: {self.prediction_mode}")
        print(f"  Run Ablation Study: {'Yes' if self.run_ablation else 'No'}")

        print("\nData Configuration:")
        print(f"  Sequence Length: {self.seq_len}")
        print(f"  Time Window: {self.time_window}s ({self.time_window/3600:.1f}h)")
        print(f"  Train/Val/Test Split: {self.train_ratio:.1%}/{self.val_ratio:.1%}/{self.test_ratio:.1%}")

        print("\nModel Architecture:")
        print(f"  d_model: {self.d_model}")
        print(f"  n_heads: {self.n_heads}")
        print(f"  e_layers: {self.e_layers}")
        print(f"  d_ff: {self.d_ff}")
        print(f"  n_primitives: {self.n_primitives}")
        print(f"  Patch: {self.use_patch} (len={self.patch_len}, stride={self.stride})")

        print("\nTraining Configuration:")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Learning Rate: {self.lr}")
        print(f"  Patience: {self.patience}")
        print(f"  Lambda_1 (MAE): {self.lambda_1}")
        print(f"  Lambda_div (Diversity): {self.lambda_div}")

        print("\n" + "=" * 80)


@dataclass
class AblationConfig:
    """Configuration for ablation studies"""
    # Ablation variants: (name, d_model, n_primitives, use_patch, description)
    variants: List[tuple] = field(default_factory=lambda: [
        ("PRISM-Full", 256, 16, True, "Full model"),
        ("PRISM-NoPatch", 256, 16, False, "Without patch embedding"),
        ("PRISM-Small", 128, 8, True, "Smaller model (d=128, prim=8)"),
        ("PRISM-Large", 512, 32, True, "Larger model (d=512, prim=32)"),
        ("PRISM-FewPrim", 256, 4, True, "Few primitives (prim=4)"),
    ])

    epochs: int = 50
    patience: int = 15
    lr: float = 0.001


# Default configurations
DEFAULT_CONFIG = ExperimentConfig()
DEFAULT_ABLATION_CONFIG = AblationConfig()


def create_custom_config(**kwargs) -> ExperimentConfig:
    """Create custom configuration with overrides"""
    return ExperimentConfig(**kwargs)


# Example usage:
if __name__ == "__main__":
    # Default configuration
    config = ExperimentConfig()
    config.print_config()

    # Custom configuration
    custom_config = create_custom_config(
        seeds=[42, 2024],
        pred_lens=[24, 48],
        gpu_ids=[0],
        prediction_mode='priority',
        d_model=512
    )
    custom_config.print_config()