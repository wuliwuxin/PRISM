"""
PRISM Optimized Training Module
Saves predictions and ground truth as .npy files for visualization
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np
import os
from metrics import MAE, MSE, RMSE, MAPE, R2


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001,
                device='cpu', patience=20, save_path='best_model.pth',
                lambda_1=0.1, lambda_div=0.01, verbose=True):
    """
    Train the PRISM model with early stopping

    Returns:
        best_epoch: Best epoch number
        best_val_loss: Best validation loss
        best_train_metrics: Training metrics at best epoch
        best_val_metrics: Validation metrics at best epoch
    """
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    best_train_metrics = None
    best_val_metrics = None

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        total_forecast_loss = 0
        total_div_loss = 0
        train_mse = 0
        train_mae = 0

        if verbose:
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]', leave=False)
        else:
            train_bar = train_loader

        for batch_x, hours, days, months, is_weekend, batch_y in train_bar:
            batch_x = batch_x.to(device)
            hours = hours.to(device)
            days = days.to(device)
            months = months.to(device)
            is_weekend = is_weekend.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions, diversity_loss_total = model(batch_x, hours, days, months, is_weekend)

            # Multi-channel regression: predictions and targets are both
            # shaped [batch, pred_len, n_channels].
            mse_loss = F.mse_loss(predictions, batch_y)
            mae_loss = F.l1_loss(predictions, batch_y)

            forecast_loss = mse_loss + lambda_1 * mae_loss
            loss = forecast_loss + lambda_div * diversity_loss_total

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            total_forecast_loss += forecast_loss.item()
            total_div_loss += diversity_loss_total.item()
            train_mse += mse_loss.item()
            train_mae += mae_loss.item()

            if verbose and hasattr(train_bar, 'set_postfix'):
                train_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mse': f'{mse_loss.item():.4f}',
                    'div': f'{diversity_loss_total.item():.4f}'
                })

        avg_train_mse = train_mse / len(train_loader)
        avg_train_mae = train_mae / len(train_loader)
        avg_train_forecast = total_forecast_loss / len(train_loader)
        avg_train_div = total_div_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_mse = 0
        val_mae = 0
        total_val_forecast = 0
        total_val_div = 0

        with torch.no_grad():
            for batch_x, hours, days, months, is_weekend, batch_y in val_loader:
                batch_x = batch_x.to(device)
                hours = hours.to(device)
                days = days.to(device)
                months = months.to(device)
                is_weekend = is_weekend.to(device)
                batch_y = batch_y.to(device)

                predictions, diversity_loss_total = model(batch_x, hours, days, months, is_weekend)

                # Multi-channel regression: predictions and targets are both
                # shaped [batch, pred_len, n_channels].
                mse_loss = F.mse_loss(predictions, batch_y)
                mae_loss = F.l1_loss(predictions, batch_y)

                forecast_loss = mse_loss + lambda_1 * mae_loss

                if not (torch.isnan(mse_loss) or torch.isinf(mse_loss)):
                    val_mse += mse_loss.item()
                    val_mae += mae_loss.item()
                    total_val_forecast += forecast_loss.item()
                    total_val_div += diversity_loss_total.item()

        avg_val_mse = val_mse / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        avg_val_forecast = total_val_forecast / len(val_loader)
        avg_val_div = total_val_div / len(val_loader)

        # Early stopping check
        if avg_val_mse < best_val_loss:
            best_val_loss = avg_val_mse
            patience_counter = 0
            # Deep-clone best weights to avoid being affected by subsequent optimizer updates.
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

            best_train_metrics = {
                'mse': avg_train_mse,
                'mae': avg_train_mae,
                'forecast_loss': avg_train_forecast,
                'div_loss': avg_train_div
            }
            best_val_metrics = {
                'mse': avg_val_mse,
                'mae': avg_val_mae,
                'forecast_loss': avg_val_forecast,
                'div_loss': avg_val_div
            }

            # Save checkpoint
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mse': best_val_loss,
            }, save_path)
        else:
            patience_counter += 1

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train[MSE: {avg_train_mse:.4f}, MAE: {avg_train_mae:.4f}] - "
                  f"Val[MSE: {avg_val_mse:.4f}, MAE: {avg_val_mae:.4f}] - "
                  f"Best: {best_val_loss:.4f} (Ep{best_epoch}) - Pat: {patience_counter}/{patience}")

        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at Epoch {epoch + 1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return best_epoch, best_val_loss, best_train_metrics, best_val_metrics


def evaluate_model(model, test_loader, dataset, device='cpu', save_predictions=True,
                   save_dir='predictions', exp_name='exp'):
    """
    Comprehensive evaluation with prediction saving

    Args:
        model: Trained model
        test_loader: Test data loader
        dataset: Dataset object (for inverse transform)
        device: Device to evaluate on
        save_predictions: Whether to save predictions as .npy files
        save_dir: Directory to save predictions
        exp_name: Experiment name for file naming

    Returns:
        Dictionary containing all evaluation metrics
    """
    model.eval()

    all_predictions_norm = []
    all_targets_norm = []
    all_predictions_original = []
    all_targets_original = []
    inference_times = []

    with torch.no_grad():
        for batch_x, hours, days, months, is_weekend, batch_y in tqdm(test_loader,
                                                                      desc='Testing',
                                                                      leave=False):
            batch_x = batch_x.to(device)
            hours = hours.to(device)
            days = days.to(device)
            months = months.to(device)
            is_weekend = is_weekend.to(device)

            start_time = time.time()
            predictions, _ = model(batch_x, hours, days, months, is_weekend)
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)

            pred_norm = predictions.cpu().numpy()  # [batch, pred_len, n_channels]
            target_norm = batch_y.cpu().numpy()   # [batch, pred_len, n_channels]

            all_predictions_norm.append(pred_norm.reshape(-1))
            all_targets_norm.append(target_norm.reshape(-1))

            # Inverse transform each channel independently.
            # Flatten across (batch, pred_len, channels) for metric computation/visualization.
            pred_original_channels = []
            target_original_channels = []
            n_channels = target_norm.shape[-1]
            for c in range(n_channels):
                pred_c = dataset.inverse_transform(pred_norm[:, :, c].reshape(-1), channel=c).flatten()
                target_c = dataset.inverse_transform(target_norm[:, :, c].reshape(-1), channel=c).flatten()
                pred_c = np.maximum(pred_c, 0)
                pred_original_channels.append(pred_c)
                target_original_channels.append(target_c)

            all_predictions_original.append(np.concatenate(pred_original_channels, axis=0))
            all_targets_original.append(np.concatenate(target_original_channels, axis=0))

    # Concatenate all predictions
    all_predictions_norm = np.concatenate(all_predictions_norm, axis=0).flatten()
    all_targets_norm = np.concatenate(all_targets_norm, axis=0).flatten()
    all_predictions_original = np.concatenate(all_predictions_original, axis=0).flatten()
    all_targets_original = np.concatenate(all_targets_original, axis=0).flatten()

    # Save predictions as .npy files
    if save_predictions:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f'{exp_name}_predictions.npy'), all_predictions_original)
        np.save(os.path.join(save_dir, f'{exp_name}_targets.npy'), all_targets_original)
        np.save(os.path.join(save_dir, f'{exp_name}_predictions_norm.npy'), all_predictions_norm)
        np.save(os.path.join(save_dir, f'{exp_name}_targets_norm.npy'), all_targets_norm)

    # Calculate metrics (normalized space)
    mse = MSE(all_predictions_norm, all_targets_norm)
    mae = MAE(all_predictions_norm, all_targets_norm)
    rmse = RMSE(all_predictions_norm, all_targets_norm)

    # Calculate metrics (original scale for interpretation)
    original_mae = MAE(all_predictions_original, all_targets_original)
    original_mse = MSE(all_predictions_original, all_targets_original)
    original_rmse = RMSE(all_predictions_original, all_targets_original)
    original_mape = MAPE(all_predictions_original, all_targets_original)
    original_r2 = R2(all_predictions_original, all_targets_original)
    avg_inference_time = np.mean(inference_times)

    return {
        # Normalized metrics (for model comparison)
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        # Original scale metrics (for interpretation)
        'mae_original': original_mae,
        'mse_original': original_mse,
        'rmse_original': original_rmse,
        'mape': original_mape,
        'r2': original_r2,
        'avg_inference_time': avg_inference_time,
        # Prediction file paths
        'predictions_file': os.path.join(save_dir, f'{exp_name}_predictions.npy') if save_predictions else None,
        'targets_file': os.path.join(save_dir, f'{exp_name}_targets.npy') if save_predictions else None
    }