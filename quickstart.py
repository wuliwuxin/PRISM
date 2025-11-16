"""
PRISM Quick Start
Minimal example to get started quickly
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from config import create_custom_config
from data_processor import GPUDataProcessor, GPUDemandDataset, prepare_datasets
from model import PRISM, count_parameters
from train import train_model, evaluate_model
from torch.utils.data import DataLoader


def quick_start_example(mode='total', pred_len=24):
    """
    Quick start example with minimal configuration

    Args:
        mode: Prediction mode ('total', 'priority', 'organization')
        pred_len: Prediction length in hours
    """
    print("=" * 80)
    print(f"PRISM Quick Start - Mode: {mode}, Prediction: {pred_len}h")
    print("=" * 80)

    # 1. Create minimal configuration
    config = create_custom_config(
        prediction_mode=mode,
        seeds=[42],  # Single seed for quick test
        pred_lens=[pred_len],
        gpu_ids=[0],  # Single GPU
        epochs=20,  # Fewer epochs for quick test
        batch_size=64,
        d_model=128,  # Smaller model
        n_primitives=8
    )

    print("\n[1/5] Configuration created")
    config.print_config()

    # 2. Load data
    print("\n[2/5] Loading data...")
    try:
        node_df = pd.read_csv('data/node_info_df.csv')
        job_df = pd.read_csv('data/job_info_df.csv')
        print(f"  ✓ Loaded: {len(node_df)} nodes, {len(job_df)} jobs")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        print("\n  Please ensure data files exist:")
        print("    - data/node_info_df.csv")
        print("    - data/job_info_df.csv")
        return

    # 3. Process data
    print("\n[3/5] Processing data...")
    processor = GPUDataProcessor(node_df, job_df)
    timeseries_data = processor.create_timeseries(mode=mode, time_window=3600)
    data_array, channel_names = prepare_datasets(
        timeseries_data, config.seq_len, pred_len, mode
    )
    print(f"  ✓ Time series shape: {data_array.shape}")
    print(f"  ✓ Channels: {channel_names}")

    # 4. Create dataset and loaders
    print("\n[4/5] Creating datasets...")
    dataset = GPUDemandDataset(data_array, config.seq_len, pred_len, mode)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=2)

    print(f"  ✓ Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # 5. Create and train model
    print("\n[5/5] Training model...")
    device = config.get_device(0)

    model = PRISM(
        seq_len=config.seq_len,
        pred_len=pred_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
        e_layers=config.e_layers,
        d_ff=config.d_model * 4,
        n_primitives=config.n_primitives,
        use_patch=True,
        patch_len=16,
        stride=8,
        dropout=0.1
    )

    total_params, _ = count_parameters(model)
    print(f"  ✓ Model created: {total_params / 1e6:.2f}M parameters")

    # Train
    os.makedirs('quickstart_output', exist_ok=True)
    save_path = f'quickstart_output/model_{mode}_{pred_len}h.pth'

    best_epoch, best_val_mse, train_metrics, val_metrics = train_model(
        model, train_loader, val_loader,
        epochs=config.epochs,
        lr=config.lr,
        device=device,
        patience=10,
        save_path=save_path,
        lambda_1=0.1,
        lambda_div=0.01,
        verbose=True
    )

    print(f"\n  ✓ Training completed!")
    print(f"    Best epoch: {best_epoch}")
    print(f"    Best val MSE: {best_val_mse:.4f}")

    # 6. Evaluate
    print("\n[6/6] Evaluating model...")
    eval_results = evaluate_model(
        model, test_loader, dataset, device=device,
        save_predictions=True,
        save_dir='quickstart_output',
        exp_name=f'quickstart_{mode}_{pred_len}h'
    )

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nTest Metrics:")
    print(f"  MAE:  {eval_results['mae']:.4f}")
    print(f"  RMSE: {eval_results['rmse']:.4f}")
    print(f"\nOriginal Scale:")
    print(f"  MAE:  {eval_results['mae_original']:.4f} GPUs")
    print(f"  RMSE: {eval_results['rmse_original']:.4f} GPUs")
    print(f"  MAPE: {eval_results['mape']:.2f}%")
    print(f"  R²:   {eval_results['r2']:.6f}")

    print(f"\nFiles saved:")
    print(f"  Model: {save_path}")
    print(f"  Predictions: {eval_results['predictions_file']}")
    print(f"  Targets: {eval_results['targets_file']}")

    # 7. Visualize
    print("\n[7/7] Creating visualization...")
    visualize_results(
        eval_results['predictions_file'],
        eval_results['targets_file'],
        f'quickstart_output/result_{mode}_{pred_len}h.png'
    )

    print("\n" + "=" * 80)
    print("✓ QUICK START COMPLETE!")
    print("=" * 80)
    print("\nCheck 'quickstart_output/' directory for results")


def visualize_results(pred_file, target_file, save_path, max_samples=500):
    """Create visualization of results"""
    predictions = np.load(pred_file)
    targets = np.load(target_file)

    # Limit samples
    predictions = predictions[:max_samples]
    targets = targets[:max_samples]

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Waveform comparison
    x = np.arange(len(predictions))
    axes[0].plot(x, targets, label='Ground Truth', color='#2E86C1', linewidth=1.5, alpha=0.8)
    axes[0].plot(x, predictions, label='Predictions', color='#E74C3C', linewidth=1.5, alpha=0.7)
    axes[0].fill_between(x, predictions, targets, alpha=0.2, color='gray')
    axes[0].set_xlabel('Time Step', fontsize=12)
    axes[0].set_ylabel('GPU Demand', fontsize=12)
    axes[0].set_title('Prediction vs Ground Truth', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Error distribution
    errors = predictions - targets
    axes[1].hist(errors, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prediction Error', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Statistics
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    axes[1].text(0.02, 0.98, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}',
                 transform=axes[1].transAxes, fontsize=11,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Visualization saved: {save_path}")


def main():
    """Main function with examples"""
    import argparse

    parser = argparse.ArgumentParser(description='PRISM Quick Start')
    parser.add_argument('--mode', type=str, default='total',
                        choices=['total', 'priority', 'organization'],
                        help='Prediction mode')
    parser.add_argument('--pred-len', type=int, default=24,
                        help='Prediction length in hours')

    args = parser.parse_args()

    # Run quick start
    quick_start_example(mode=args.mode, pred_len=args.pred_len)


if __name__ == "__main__":
    main()