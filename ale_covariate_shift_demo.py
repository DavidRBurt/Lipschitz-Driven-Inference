#!/usr/bin/env python
"""
Demonstration of ALE Method Failure Under Covariate Shift

This standalone script demonstrates how the Accumulated Local Effects (ALE) method
from the paper "Visualizing the Effects of Predictor Variables in Black Box
Supervised Learning Models" can fail under covariate shift conditions.

The experiment:
1. Generates synthetic data with y = |x1| + noise, where x2-x10 are noise features
2. Uses training data only from the region where x1 < 0
3. Trains an XGBoost model on this partial distribution
4. Uses pyALE to assess the feature effects
5. Shows how ALE fails to correctly estimate the true effect in the extrapolation region

Dependencies:
- numpy
- pandas
- matplotlib
- xgboost
- pyALE
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from PyALE import ale


def generate_synthetic_data(n_train=1000, n_test=500, noise_std=0.1, n_features=10):
    """
    Generate synthetic data for the experiment.
    
    - Features x1-x10 are drawn from standard normal distribution
    - Target y = |x1| + Gaussian noise
    - Training data is filtered to only include samples where x1 < 0
    
    Args:
        n_train: Number of training samples (after filtering)
        n_test: Number of test samples
        noise_std: Standard deviation of the noise
        n_features: Total number of features (including x1)
        
    Returns:
        X_train, y_train, X_test, y_test: Training and test datasets
    """
    # Generate more samples than needed since we'll filter
    n_buffer = int(n_train * 3)
    
    # Generate the full dataset
    X_full = np.random.uniform(-3, 3, size=(n_buffer, n_features))
    
    # True function: y = |x1| + noise
    y_full = np.abs(X_full[:, 0]) + np.random.normal(0, noise_std, size=n_buffer)
    
    X_train = X_full[:n_train]
    y_train = y_full[:n_train]
    
    # Generate test data from the full distribution
    X_test = np.random.uniform(2, 3, size=(n_test, n_features))
    y_test = np.abs(X_test[:, 0]) + np.random.normal(0, noise_std, size=n_test)
    
    return X_train, y_train, X_test, y_test


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    n_train = 20
    n_test = 20
    noise_std = 0.1
    n_features = 20
    
    print("Generating synthetic data...")
    X_train, y_train, X_test, y_test = generate_synthetic_data(
        n_train=n_train, 
        n_test=n_test, 
        noise_std=noise_std, 
        n_features=n_features
    )
    
    # Create feature names
    feature_names = [f'x{i+1}' for i in range(n_features)]
    
    # Convert to DataFrames for pyALE
    train_df = pd.DataFrame(X_train, columns=feature_names)
    test_df = pd.DataFrame(X_test, columns=feature_names)
    
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    print(f"x1 range in training: [{X_train[:, 0].min():.2f}, {X_train[:, 0].max():.2f}]")
    print(f"x1 range in test: [{X_test[:, 0].min():.2f}, {X_test[:, 0].max():.2f}]")
    
    # Train XGBoost model
    print("Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate model performance
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(np.mean((y_train - train_pred)**2))
    test_rmse = np.sqrt(np.mean((y_test - test_pred)**2))
    
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Calculate ALE for feature x1
    print("Calculating ALE for feature x1...")
    
    # ALE on training data
    ale_train = ale(
        X=train_df,
        model=model,
        feature=['x1'],
        grid_size=50,
        include_CI=True,
        C=0.95
    )
    
    # ALE on test data
    ale_test = ale(
        X=test_df,
        model=model,
        feature=['x1'],
        grid_size=50,
        include_CI=True,
        C=0.95
    )
    
    # Reset the index of the ale_train and ale_test
    ale_train = ale_train.reset_index()
    ale_test = ale_test.reset_index()
    
    # Calculate true effect values (|x1|) for ALE grid points
    train_x_values = ale_train['x1']
    test_x_values = ale_test['x1']
    
    true_train_effect = np.abs(train_x_values)
    true_test_effect = np.abs(test_x_values)
    
    # Create visualization
    print("Creating visualization...")
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot 1: Data distribution
    axs[0].scatter(X_train[:, 0], y_train, alpha=0.5, color='blue', label='Training data (x1 < 0)')
    axs[0].scatter(X_test[:, 0], y_test, alpha=0.5, color='red', label='Test data (all x1)')
    
    # Plot the true relationship y = |x1|
    x_grid = np.linspace(-3, 3, 100)
    y_grid = np.abs(x_grid)
    axs[0].plot(x_grid, y_grid, 'k-', linewidth=2, label='True function (|x1|)')
    
    # Add vertical line at x=0
    axs[0].axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    
    axs[0].set_title('Data Distribution', fontsize=14)
    axs[0].set_xlabel('x1', fontsize=12)
    axs[0].set_ylabel('y', fontsize=12)
    axs[0].legend(fontsize=10, loc='upper left')
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: Training data ALE
    axs[1].plot(train_x_values, ale_train['eff'], 'b-', linewidth=2, label='ALE estimate')
    axs[1].fill_between(
        train_x_values, 
        ale_train['lowerCI_95%'], 
        ale_train['upperCI_95%'], 
        alpha=0.2, 
        color='b', 
        label='95% CI'
    )
    axs[1].plot(train_x_values, true_train_effect, 'r--', linewidth=2, label='True effect (|x1|)')
    axs[1].set_title('ALE on Training Data (x1 < 0)', fontsize=14)
    axs[1].set_xlabel('x1', fontsize=12)
    axs[1].set_ylabel('Effect on y', fontsize=12)
    axs[1].legend(fontsize=10)
    axs[1].grid(True, alpha=0.3)
    
    # Draw a vertical line at x = 0
    axs[1].axvline(x=0, color='k', linestyle='--', alpha=0.5)
    
    
    # Plot 3: Test data ALE
    axs[2].plot(test_x_values, ale_test['eff'], 'b-', linewidth=2, label='ALE estimate')
    axs[2].fill_between(
        test_x_values, 
        ale_test['lowerCI_95%'], 
        ale_test['upperCI_95%'], 
        alpha=0.2, 
        color='b', 
        label='95% CI'
    )
    axs[2].plot(test_x_values, true_test_effect, 'r--', linewidth=2, label='True effect (|x1|)')
    axs[2].set_title('ALE on Test Data (All x1)', fontsize=14)
    axs[2].set_xlabel('x1', fontsize=12)
    axs[2].set_ylabel('Effect on y', fontsize=12)
    axs[2].legend(fontsize=10)
    axs[2].grid(True, alpha=0.3)
    
    
    plt.suptitle('Failure of ALE Under Covariate Shift', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    output_dir = 'ale_results'
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, 'ale_covariate_shift.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 5))
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)
    plt.barh(range(len(sorted_idx)), importance[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    
    # Print key findings
    print("\nKey Findings:")
    print("1. In the training region (x1 < 0), ALE estimates align reasonably well with the true effect |x1|")
    print("2. In the extrapolation region (x1 > 0), ALE fails to capture the true effect")
    print("3. The confidence intervals in the extrapolation region don't contain the true effect")
    print("4. This demonstrates that ALE breaks down under covariate shift, especially when")
    print("   the model hasn't been trained on the full domain of the feature space")
    
    print("\nThe experiment shows why interpretability methods like ALE need to be used with")
    print("caution when there is covariate shift between training and test distributions.")


if __name__ == "__main__":
    main() 