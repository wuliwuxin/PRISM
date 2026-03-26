"""
PRISM Main Experiment Script - FIXED VERSION
Replace your main_optimized.py with this file
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import time
import warnings
import os
import sys
import itertools
import argparse

warnings.filterwarnings('ignore')

from config import ExperimentConfig, AblationConfig, create_custom_config
from data_processor import GPUDataProcessor, GPUDemandDataset, prepare_datasets
from model import PRISM, count_parameters
from train import train_model, evaluate_model

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(dataset, config: ExperimentConfig, seed: int):
    """Create train/val/test dataloaders"""
    total_size = len(dataset)
    train_size = int(config.train_ratio * total_size)
    val_size = int(config.val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, (train_size, val_size, test_size)


def run_single_experiment(data_array, channel_names, seed, pred_len,
                          config: ExperimentConfig, device, experiment_id):
    """Run a single experiment"""
    set_seed(seed)

    print(f"\n{'=' * 80}")
    print(f"Experiment {experiment_id}: seed={seed}, pred_len={pred_len}, mode={config.prediction_mode}")
    print(f"{'=' * 80}")

    # Create dataset
    dataset = GPUDemandDataset(
        data_array,
        config.seq_len,
        pred_len,
        config.prediction_mode,
        time_window_seconds=config.time_window
    )

    # Create dataloaders
    train_loader, val_loader, test_loader, sizes = create_dataloaders(dataset, config, seed)
    train_size, val_size, test_size = sizes

    print(f"Dataset: train={train_size}, val={val_size}, test={test_size}")
    print(f"Channels: {channel_names}")

    # Create model
    model = PRISM(
        seq_len=config.seq_len,
        pred_len=pred_len,
        n_channels=len(channel_names),
        use_patch=config.use_patch,
        patch_len=config.patch_len,
        stride=config.stride,
        d_model=config.d_model,
        n_heads=config.n_heads,
        e_layers=config.e_layers,
        d_ff=config.d_ff,
        n_primitives=config.n_primitives,
        dropout=config.dropout
    )

    total_params, trainable_params = count_parameters(model)
    print(f"Parameters: {total_params / 1e6:.2f}M total, {trainable_params / 1e6:.2f}M trainable")

    # Save path
    exp_name = f"prism_{config.prediction_mode}_seed{seed}_predlen{pred_len}"
    save_path = os.path.join(config.checkpoint_dir, f'{exp_name}.pth')

    # Train - FIXED: Don't pass batch_size here
    train_start = time.time()
    best_epoch, best_val_mse, best_train_metrics, best_val_metrics = train_model(
        model, train_loader, val_loader,
        epochs=config.epochs,
        lr=config.lr,
        device=device,
        patience=config.patience,
        save_path=save_path,
        lambda_1=config.lambda_1,
        lambda_div=config.lambda_div,
        verbose=(experiment_id == 1)
    )
    train_time = time.time() - train_start

    # Evaluate
    eval_results = evaluate_model(
        model, test_loader, dataset, device=device,
        save_predictions=True,
        save_dir=config.predictions_dir,
        exp_name=exp_name
    )

    # Print results
    print(f"\n--- Test Results ---")
    print(f"  MAE:  {eval_results['mae']:.4f}")
    print(f"  MSE:  {eval_results['mse']:.4f}")
    print(f"  RMSE: {eval_results['rmse']:.4f}")

    print(f"\n--- Original Scale ---")
    print(f"  MAE:  {eval_results['mae_original']:.4f} GPUs")
    print(f"  RMSE: {eval_results['rmse_original']:.4f} GPUs")
    print(f"  MAPE: {eval_results['mape']:.2f}%")
    print(f"  R²:   {eval_results['r2']:.6f}")

    print(f"\n--- Performance ---")
    print(f"  Training time: {train_time / 60:.2f} min")
    print(f"  Inference time: {eval_results['avg_inference_time']:.2f} ms/batch")
    print(f"  Best epoch: {best_epoch}")

    # Return results - FIXED: Explicit field names
    result = {
        'experiment_id': experiment_id,
        'seed': seed,
        'pred_len': pred_len,
        'prediction_mode': config.prediction_mode,
        'channels': ','.join(channel_names),
        # Model config
        'seq_len': config.seq_len,
        'd_model': config.d_model,
        'n_heads': config.n_heads,
        'e_layers': config.e_layers,
        'd_ff': config.d_ff,
        'n_primitives': config.n_primitives,
        'use_patch': config.use_patch,
        'patch_len': config.patch_len,
        'stride': config.stride,
        'dropout': config.dropout,
        # Training config
        'batch_size': config.batch_size,
        'epochs': config.epochs,
        'learning_rate': config.lr,
        'patience': config.patience,
        'lambda_1': config.lambda_1,
        'lambda_div': config.lambda_div,
        # Training metrics
        'train_mse': best_train_metrics['mse'],
        'train_mae': best_train_metrics['mae'],
        'val_mse': best_val_metrics['mse'],
        'val_mae': best_val_metrics['mae'],
        # Test metrics (normalized)
        'test_mse': eval_results['mse'],
        'test_mae': eval_results['mae'],
        'test_rmse': eval_results['rmse'],
        # Test metrics (original)
        'test_mae_original': eval_results['mae_original'],
        'test_rmse_original': eval_results['rmse_original'],
        'test_mape': eval_results['mape'],
        'test_r2': eval_results['r2'],
        # Performance
        'training_time_min': train_time / 60,
        'inference_time_ms': eval_results['avg_inference_time'],
        'parameters_M': total_params / 1e6,
        'best_epoch': best_epoch,
        # Files
        'model_path': save_path,
        'predictions_file': eval_results['predictions_file'],
        'targets_file': eval_results['targets_file']
    }

    return result


def run_ablation_study(data_array, channel_names, config: ExperimentConfig,
                      ablation_config: AblationConfig, device):
    """Run ablation study"""
    print("\n" + "=" * 80)
    print("ABLATION STUDY")
    print("=" * 80)

    set_seed(42)
    pred_len = 24

    dataset = GPUDemandDataset(
        data_array,
        config.seq_len,
        pred_len,
        config.prediction_mode,
        time_window_seconds=config.time_window
    )
    train_loader, val_loader, test_loader, _ = create_dataloaders(dataset, config, 42)

    ablation_results = []

    for variant_name, d_model, n_primitives, use_patch, description in ablation_config.variants:
        print(f"\n{'=' * 80}")
        print(f"{variant_name}: {description}")
        print(f"{'=' * 80}")

        model = PRISM(
            seq_len=config.seq_len,
            pred_len=pred_len,
            n_channels=len(channel_names),
            use_patch=use_patch,
            patch_len=16,
            stride=8,
            d_model=d_model,
            n_heads=8,
            e_layers=3,
            d_ff=d_model * 4,
            n_primitives=n_primitives,
            dropout=0.1
        )

        total_params, _ = count_parameters(model)
        print(f"Parameters: {total_params / 1e6:.2f}M")

        save_path = os.path.join(config.checkpoint_dir, f'ablation_{variant_name}.pth')

        train_start = time.time()
        best_epoch, _ = train_model(
            model, train_loader, val_loader,
            epochs=ablation_config.epochs,
            lr=ablation_config.lr,
            device=device,
            patience=ablation_config.patience,
            save_path=save_path,
            lambda_1=1,
            lambda_div=1,
            # lambda_1=0.1,
            # lambda_div=0.01,
            verbose=False
        )
        train_time = time.time() - train_start

        exp_name = f'ablation_{variant_name}'
        eval_results = evaluate_model(
            model, test_loader, dataset, device=device,
            save_predictions=True,
            save_dir=config.predictions_dir,
            exp_name=exp_name
        )

        print(f"  MAE: {eval_results['mae_original']:.4f} GPUs, R²: {eval_results['r2']:.6f}")

        ablation_results.append({
            'config_name': variant_name,
            'description': description,
            'd_model': d_model,
            'n_primitives': n_primitives,
            'use_patch': use_patch,
            'parameters_M': total_params / 1e6,
            'test_mae': eval_results['mae'],
            'test_mae_original': eval_results['mae_original'],
            'test_rmse_original': eval_results['rmse_original'],
            'test_mape': eval_results['mape'],
            'test_r2': eval_results['r2'],
            'training_time_min': train_time / 60,
            'predictions_file': eval_results['predictions_file']
        })

    return ablation_results


def main(custom_config=None):
    """Main experiment pipeline"""
    print("=" * 80)
    print("PRISM: Optimized Experiment Pipeline")
    print("=" * 80)

    # Load configuration
    if custom_config is None:
        config = ExperimentConfig()
    else:
        config = custom_config

    config.print_config()

    # Create directories
    for dir_name in [config.checkpoint_dir, config.results_dir, config.visualization_dir,
                     config.predictions_dir, config.log_dir]:
        os.makedirs(dir_name, exist_ok=True)

    # Load data
    print("\n[1/4] Loading data...")
    try:
        node_df = pd.read_csv(os.path.join(config.data_dir, 'node_info_df.csv'))
        job_df = pd.read_csv(os.path.join(config.data_dir, 'job_info_df.csv'))
        print(f"  ✓ Node: {len(node_df)} rows, Job: {len(job_df)} rows")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        sys.exit(1)

    # Generate time series
    print("\n[2/4] Generating time series...")
    processor = GPUDataProcessor(node_df, job_df)
    timeseries_data = processor.create_timeseries(
        mode=config.prediction_mode,
        time_window=config.time_window
    )

    # Prepare datasets
    data_array, channel_names = prepare_datasets(
        timeseries_data, config.seq_len,
        config.pred_lens[0], config.prediction_mode
    )

    print(f"  ✓ Length: {len(data_array)}")
    print(f"  ✓ Channels: {channel_names}")
    print(f"  ✓ Shape: {data_array.shape}")

    # GPU configuration
    print(f"\n[3/4] GPU Configuration:")
    for gpu_id in config.gpu_ids:
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        else:
            print(f"  GPU {gpu_id}: Not available")

    # Run main experiments
    print(f"\n[4/4] Running main experiments...")
    experiments = list(itertools.product(config.seeds, config.pred_lens))
    total_experiments = len(experiments)
    print(f"  Total: {total_experiments} experiments")
    print(f"  Seeds: {config.seeds}")
    print(f"  Pred lengths: {config.pred_lens}")

    results = []

    for exp_idx, (seed, pred_len) in enumerate(experiments, 1):
        # Assign GPU
        gpu_id = config.gpu_ids[exp_idx % len(config.gpu_ids)]
        device = config.get_device(gpu_id)

        try:
            result = run_single_experiment(
                data_array, channel_names, seed, pred_len,
                config, device, exp_idx
            )
            results.append(result)

            # Save intermediate results
            if len(results) > 0:
                results_df = pd.DataFrame(results)
                results_file = os.path.join(config.results_dir,
                                          f'prism_{config.prediction_mode}_results.csv')
                results_df.to_csv(results_file, index=False)

        except Exception as e:
            print(f"\n✗ Experiment {exp_idx} failed: {e}")
            import traceback
            traceback.print_exc()

    # Run ablation study (optional)
    if len(results) > 0 and config.run_ablation:
        print(f"\n[5/5] Running ablation study...")
        device = config.get_device(config.gpu_ids[0])
        ablation_config = AblationConfig()

        try:
            ablation_results = run_ablation_study(
                data_array, channel_names, config, ablation_config, device
            )

            if len(ablation_results) > 0:
                ablation_df = pd.DataFrame(ablation_results)
                ablation_file = os.path.join(config.results_dir,
                                            f'prism_{config.prediction_mode}_ablation.csv')
                ablation_df.to_csv(ablation_file, index=False)
                print(f"\n✓ Ablation results saved: {ablation_file}")
        except Exception as e:
            print(f"\n✗ Ablation failed: {e}")
    elif config.run_ablation:
        print("\n⚠ Skipping ablation study (no main results)")
    else:
        print("\n⏭ Skipping ablation study (disabled)")

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    if len(results) > 0:
        results_df = pd.DataFrame(results)

        summary = results_df.groupby('pred_len').agg({
            'test_mae_original': ['mean', 'std', 'min', 'max'],
            'test_rmse_original': ['mean', 'std', 'min', 'max'],
            'test_mape': ['mean', 'std'],
            'test_r2': ['mean', 'std'],
        }).round(4)

        print("\nBy Prediction Length:")
        print(summary)

        best_results = results_df.loc[results_df.groupby('pred_len')['test_mae_original'].idxmin()]
        print("\nBest for each pred_len:")
        print(best_results[['pred_len', 'seed', 'test_mae_original',
                           'test_rmse_original', 'test_mape', 'test_r2']].to_string(index=False))

        overall_best = results_df.loc[results_df['test_mae_original'].idxmin()]
        print(f"\n--- OVERALL BEST ---")
        print(f"Seed: {overall_best['seed']}, Pred: {overall_best['pred_len']}h")
        print(f"MAE: {overall_best['test_mae_original']:.4f} GPUs")
        print(f"RMSE: {overall_best['test_rmse_original']:.4f} GPUs")
        print(f"MAPE: {overall_best['test_mape']:.2f}%")
        print(f"R²: {overall_best['test_r2']:.6f}")
        print(f"Predictions: {overall_best['predictions_file']}")

    print("\n" + "=" * 80)
    print("✓ COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PRISM Experiments')
    parser.add_argument('--mode', type=str, default='total',
                       choices=['total', 'priority', 'organization'],
                       help='Prediction mode')
    parser.add_argument('--seeds', type=int, nargs='+',
                       default=[42, 2024, 123456, 2025, 2026],
                       help='Random seeds')
    parser.add_argument('--pred_lens', type=int, nargs='+',
                       default=[6, 12, 24, 48],
                       help='Prediction lengths')
    parser.add_argument('--gpus', type=int, nargs='+',
                       default=[0, 1],
                       help='GPU IDs')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--ablation', action='store_true',
                       help='Run ablation study (default: False)')

    args = parser.parse_args()

    # Create custom configuration
    custom_config = create_custom_config(
        prediction_mode=args.mode,
        seeds=args.seeds,
        pred_lens=args.pred_lens,
        gpu_ids=args.gpus,
        epochs=args.epochs,
        batch_size=args.batch_size,
        run_ablation=args.ablation
    )

    try:
        main(custom_config)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)