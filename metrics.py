"""
Evaluation Metrics for Time Series Forecasting
Complete set of metrics for model evaluation
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def MAE(pred, true):
    """Mean Absolute Error"""
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """Mean Squared Error"""
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """Root Mean Squared Error"""
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true, epsilon=1e-8):
    """
    Mean Absolute Percentage Error
    Add epsilon to avoid division by zero
    """
    mask = np.abs(true) > epsilon
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((true[mask] - pred[mask]) / true[mask])) * 100


def MSPE(pred, true, epsilon=1e-8):
    """Mean Squared Percentage Error"""
    mask = np.abs(true) > epsilon
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.square((true[mask] - pred[mask]) / true[mask])) * 100


def SMAPE(pred, true, epsilon=1e-8):
    """
    Symmetric Mean Absolute Percentage Error
    Range: 0-200%, symmetric for predictions and true values
    """
    numerator = np.abs(pred - true)
    denominator = (np.abs(true) + np.abs(pred)) / 2
    mask = denominator > epsilon
    if mask.sum() == 0:
        return 0.0
    return np.mean(numerator[mask] / denominator[mask]) * 100


def RSE(pred, true):
    """
    Root Relative Squared Error
    Measures improvement over mean baseline
    """
    numerator = np.sqrt(np.sum((true - pred) ** 2))
    denominator = np.sqrt(np.sum((true - true.mean()) ** 2))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def RAE(pred, true):
    """Relative Absolute Error"""
    numerator = np.sum(np.abs(true - pred))
    denominator = np.sum(np.abs(true - true.mean()))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def CORR(pred, true):
    """
    Correlation Coefficient
    Measures linear correlation between predictions and true values
    """
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2).sum(0) * ((pred - pred.mean(0)) ** 2).sum(0))
    if d == 0:
        return 0.0
    return (u / d).mean() if hasattr(u / d, 'mean') else (u / d)


def R2(pred, true):
    """
    R-squared / Coefficient of Determination
    Range: (-∞, 1], 1 indicates perfect prediction
    """
    return r2_score(true, pred)


def NRMSE(pred, true):
    """
    Normalized RMSE
    Normalized by the range of true values
    """
    rmse = RMSE(pred, true)
    value_range = true.max() - true.min()
    if value_range == 0:
        return 0.0
    return rmse / value_range


def WAPE(pred, true, epsilon=1e-8):
    """
    Weighted Absolute Percentage Error
    Improved version of MAPE using sum instead of mean
    """
    numerator = np.sum(np.abs(true - pred))
    denominator = np.sum(np.abs(true))
    if denominator < epsilon:
        return 0.0
    return (numerator / denominator) * 100


def MdAE(pred, true):
    """
    Median Absolute Error
    More robust to outliers
    """
    return np.median(np.abs(pred - true))


def MASE(pred, true, seasonality=1):
    """
    Mean Absolute Scaled Error
    Scale-free metric, compares to naive forecast
    """
    n = len(true)
    if n <= seasonality:
        return MAE(pred, true)

    # Naive forecast error
    naive_error = np.mean(np.abs(true[seasonality:] - true[:-seasonality]))
    if naive_error == 0:
        return 0.0

    mae = MAE(pred, true)
    return mae / naive_error


def quantile_loss(pred, true, quantile=0.5):
    """
    Quantile Loss
    Used for probabilistic forecast evaluation
    """
    errors = true - pred
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))


def metric(pred, true, detailed=False):
    """
    Calculate all major evaluation metrics

    Args:
        pred: Prediction array
        true: True value array
        detailed: Whether to return detailed metrics

    Returns:
        If detailed=False: (mae, mse, rmse, mape, mspe)
        If detailed=True: Dictionary containing all metrics
    """
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    if not detailed:
        return mae, mse, rmse, mape, mspe

    # Detailed metrics
    return {
        # Basic metrics
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'MSPE': mspe,

        # Extended metrics
        'SMAPE': SMAPE(pred, true),
        'R2': R2(pred, true),
        'RSE': RSE(pred, true),
        'RAE': RAE(pred, true),
        'CORR': CORR(pred, true),
        'NRMSE': NRMSE(pred, true),
        'WAPE': WAPE(pred, true),
        'MdAE': MdAE(pred, true),
        'MASE': MASE(pred, true)
    }


def print_metrics(metrics_dict, title="Evaluation Metrics"):
    """
    Print evaluation metrics in formatted way

    Args:
        metrics_dict: Metrics dictionary
        title: Title string
    """
    print(f"\n{'=' * 60}")
    print(f"{title}")
    print(f"{'=' * 60}")

    # Basic metrics
    basic_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE']
    print("\nBasic Metrics:")
    for key in basic_metrics:
        if key in metrics_dict:
            value = metrics_dict[key]
            if key in ['MAPE', 'MSPE']:
                print(f"  {key:8s}: {value:10.2f}%")
            else:
                print(f"  {key:8s}: {value:10.6f}")

    # Extended metrics
    extended_metrics = ['SMAPE', 'R2', 'RSE', 'RAE', 'CORR', 'NRMSE', 'WAPE', 'MdAE', 'MASE']
    if any(key in metrics_dict for key in extended_metrics):
        print("\nExtended Metrics:")
        for key in extended_metrics:
            if key in metrics_dict:
                value = metrics_dict[key]
                if key in ['SMAPE', 'WAPE']:
                    print(f"  {key:8s}: {value:10.2f}%")
                else:
                    print(f"  {key:8s}: {value:10.6f}")

    print(f"{'=' * 60}\n")


# Test code
if __name__ == "__main__":
    # Generate test data
    np.random.seed(42)
    true_values = np.random.randn(1000) * 10 + 100
    pred_values = true_values + np.random.randn(1000) * 2

    # Calculate all metrics
    metrics_dict = metric(pred_values, true_values, detailed=True)

    # Print results
    print_metrics(metrics_dict, "Test Metrics Results")

    # Simple test
    print("\nSimple Test (5 basic metrics):")
    mae, mse, rmse, mape, mspe = metric(pred_values, true_values)
    print(f"MAE:  {mae:.6f}")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"MSPE: {mspe:.2f}%")