#!/usr/bin/env python
"""
Demonstration of ALE Method Failure Under Spatial Covariate Shift

This script demonstrates how the Accumulated Local Effects (ALE) method
can fail under spatial covariate shift, where the training data covers only
a specific region of the input space and the test data comes from another region.

The experiment:
1. Generates synthetic spatial data where y depends non-linearly on x1 and x2 coordinates
2. Restricts training data to a specific square region in the 2D space
3. Generates test data from a corner subset of the training region
4. Trains an XGBoost model on the training data
5. Uses pyALE to assess feature effects on both training and test data
6. Shows how ALE fails to correctly estimate effects when distribution shifts

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
from matplotlib import cm
import xgboost as xgb
from PyALE import ale


def spatial_function(X):
    """
    A complex spatial function that depends on x1 and x2 coordinates.
    
    Args:
        X: Feature matrix where first two columns are spatial coordinates
    
    Returns:
        y: Target values
    """
    # Extract spatial coordinates
    x1 = X[:, 0]
    x2 = X[:, 1]
    
    # Create a non-linear spatial function:
    # A combination of a peak and a valley with some sinusoidal patterns
    r = np.sqrt((x1 - 0.5)**2 + (x2 - 0.5)**2)  # Distance from center point (0.5, 0.5)
    z1 = 2 * np.exp(-8 * r**2)  # Peak at center
    z2 = 0.5 * np.sin(2 * np.pi * x1) * np.cos(2 * np.pi * x2)  # Oscillation pattern
    z3 = 0.3 * (x1 * x2)  # Interaction term
    
    return z1 + z2 + z3


def generate_spatial_data(n_train=200, n_test=100, noise_std=0.05, n_features=10):
    """
    Generate synthetic spatial data with training and test data from different regions.
    
    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        noise_std: Standard deviation of Gaussian noise
        n_features: Total number of features (including x1 and x2)
    
    Returns:
        X_train, y_train, X_test, y_test: Training and test data
    """
    # Generate training data from the square region [0.1, 0.9] x [0.1, 0.9]
    X_train = np.random.uniform(0.1, 0.9, size=(n_train, n_features))
    
    # Generate test data from a subset of the training region [0.6, 0.8] x [0.6, 0.8]
    # This ensures test data is contained within the training region
    X_test = np.random.uniform(0.6, 0.8, size=(n_test, n_features))
    
    # The first two features (x1, x2) are our spatial coordinates
    # The rest are noise variables
    
    # Generate target values using the spatial function plus noise
    y_train = spatial_function(X_train) + np.random.normal(0, noise_std, size=n_train)
    y_test = spatial_function(X_test) + np.random.normal(0, noise_std, size=n_test)
    
    return X_train, y_train, X_test, y_test


def visualize_spatial_function(resolution=100, ax=None):
    """
    Create a visualization of the true spatial function.
    
    Args:
        resolution: Number of points in each dimension for the grid
        ax: Matplotlib axis to plot on
    
    Returns:
        ax: The matplotlib axis with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a grid of points
    x1_grid = np.linspace(0, 1, resolution)
    x2_grid = np.linspace(0, 1, resolution)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)
    
    # Create a feature matrix for the grid points
    X_grid = np.zeros((resolution**2, 3))
    X_grid[:, 0] = X1.flatten()
    X_grid[:, 1] = X2.flatten()
    
    # Calculate function values
    Z = spatial_function(X_grid).reshape(resolution, resolution)
    
    # Plot the surface
    im = ax.imshow(Z, extent=[0, 1, 0, 1], origin='lower', cmap=cm.viridis)
    plt.colorbar(im, ax=ax, label='f(x1, x2)')
    
    # Add contour lines for better visualization
    contour = ax.contour(X1, X2, Z, colors='k', alpha=0.3)
    
    # Highlight regions
    training_rect = plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, edgecolor='blue', 
                                 linestyle='--', linewidth=2, label='Training region')
    test_rect = plt.Rectangle((0.6, 0.6), 0.2, 0.2, fill=False, edgecolor='red',
                             linestyle='--', linewidth=2, label='Test region')
    ax.add_patch(training_rect)
    ax.add_patch(test_rect)
    
    # Set labels and title
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('True Spatial Function')
    ax.legend()
    
    return ax


def calculate_true_marginal_effect(x1_values, x2_value, n_features=10, delta=0.001):
    """
    Calculate the true marginal effect of x1 at a given x2 value.
    
    Args:
        x1_values: Array of x1 values to calculate effect for
        x2_value: Fixed x2 value
        n_features: Total number of features
        delta: Small increment for finite difference approximation
        
    Returns:
        Array of true marginal effects
    """
    X_grid = np.zeros((len(x1_values), n_features))
    X_grid[:, 0] = x1_values
    X_grid[:, 1] = x2_value
    
    # Calculate finite difference approximation of partial derivative
    X_grid_plus = X_grid.copy()
    X_grid_plus[:, 0] += delta
    
    y_grid = spatial_function(X_grid)
    y_grid_plus = spatial_function(X_grid_plus)
    
    return (y_grid_plus - y_grid) / delta


def calculate_average_marginal_effect(x1_values, x2_range=(0.1, 0.9), n_x2_points=20, n_features=10):
    """
    Calculate the average marginal effect of x1 across a range of x2 values.
    
    Args:
        x1_values: Array of x1 values to calculate effect for
        x2_range: Tuple with min and max x2 values to average over
        n_x2_points: Number of x2 points to use for averaging
        n_features: Total number of features
        
    Returns:
        Array of average marginal effects
    """
    # Generate a grid of x2 values
    x2_values = np.linspace(x2_range[0], x2_range[1], n_x2_points)
    
    # Initialize array to store all marginal effects
    all_effects = np.zeros((n_x2_points, len(x1_values)))
    
    # Calculate marginal effect for each x2 value
    for i, x2 in enumerate(x2_values):
        all_effects[i, :] = calculate_true_marginal_effect(x1_values, x2, n_features)
    
    # Return the average effect across all x2 values
    return np.mean(all_effects, axis=0)


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    n_train = 500
    n_test = 200
    noise_std = 0.05
    n_features = 10  # 2 spatial coordinates + 8 noise features
    
    print("Generating synthetic spatial data...")
    X_train, y_train, X_test, y_test = generate_spatial_data(
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
    print(f"x2 range in training: [{X_train[:, 1].min():.2f}, {X_train[:, 1].max():.2f}]")
    print(f"x2 range in test: [{X_test[:, 1].min():.2f}, {X_test[:, 1].max():.2f}]")
    
    # Calculate average x2 values for both training and test sets
    avg_x2_train = np.mean(X_train[:, 1])
    avg_x2_test = np.mean(X_test[:, 1])
    
    print(f"Average x2 in training: {avg_x2_train:.2f}")
    print(f"Average x2 in test: {avg_x2_test:.2f}")
    
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
    print("Calculating ALE for x1...")
    
    # ALE on training data
    ale_train_x1 = ale(
        X=train_df,
        model=model,
        feature=['x1'],
        grid_size=50,
        include_CI=True,
        C=0.95
    )
    
    # ALE on test data
    ale_test_x1 = ale(
        X=test_df,
        model=model,
        feature=['x1'],
        grid_size=50,
        include_CI=True,
        C=0.95
    )
    
    # Reset index for easier plotting
    ale_train_x1 = ale_train_x1.reset_index()
    ale_test_x1 = ale_test_x1.reset_index()
    
    # Create the multi-panel visualization
    print("Creating visualization...")
    fig = plt.figure(figsize=(20, 12))
    
    # Define a grid for the subplots
    grid = plt.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 1], hspace=0.3, wspace=0.3)
    
    # Plot 1: True spatial function and data points
    ax1 = fig.add_subplot(grid[0, 0])
    visualize_spatial_function(resolution=100, ax=ax1)
    
    # Add scatter points for training and test data
    ax1.scatter(X_train[:, 0], X_train[:, 1], c='blue', s=10, alpha=0.5, label='Training data')
    ax1.scatter(X_test[:, 0], X_test[:, 1], c='red', s=10, alpha=0.5, label='Test data')
    ax1.legend()
    
    # Plot 2: Data distributions
    ax2 = fig.add_subplot(grid[0, 1])
    # Plot the 1D distribution of x1 for both datasets
    ax2.hist(X_train[:, 0], bins=20, alpha=0.5, color='blue', label='Training data')
    ax2.hist(X_test[:, 0], bins=20, alpha=0.5, color='red', label='Test data')
    ax2.set_xlabel('x1 value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of x1 in Training vs. Test Data')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: ALE for training data with true effect
    ax3 = fig.add_subplot(grid[1, 0])
    ax3.plot(ale_train_x1['x1'], ale_train_x1['eff'], 'b-', linewidth=2, label='ALE estimate')
    ax3.fill_between(
        ale_train_x1['x1'],
        ale_train_x1['lowerCI_95%'],
        ale_train_x1['upperCI_95%'],
        alpha=0.2,
        color='b',
        label='95% CI'
    )
    
    # Calculate and add true marginal effect for the training range 
    x1_grid_train = np.linspace(min(ale_train_x1['x1']), max(ale_train_x1['x1']), 100)
    
    # Calculate effect at average x2
    true_effect_avg_x2 = calculate_true_marginal_effect(x1_grid_train, avg_x2_train, n_features)
    true_effect_avg_x2_scaled = true_effect_avg_x2 - np.mean(true_effect_avg_x2)
    
    # Calculate average effect over all x2 values
    true_effect_all_x2 = calculate_average_marginal_effect(
        x1_grid_train, 
        x2_range=(X_train[:, 1].min(), X_train[:, 1].max()),
        n_x2_points=20,
        n_features=n_features
    )
    true_effect_all_x2_scaled = true_effect_all_x2 - np.mean(true_effect_all_x2)
    
    # Add both true effects to the plot
    ax3.plot(x1_grid_train, true_effect_avg_x2_scaled, 'g--', linewidth=1.5, 
             label=f'True effect at mean x2={avg_x2_train:.2f}')
    ax3.plot(x1_grid_train, true_effect_all_x2_scaled, 'r--', linewidth=2, 
             label='True effect averaged over all x2')
    
    # Add vertical lines to mark the test data region
    ax3.axvspan(0.6, 0.8, alpha=0.1, color='r', label='Test data region')
    ax3.set_title('ALE for x1 on Training Data', fontsize=14)
    ax3.set_xlabel('x1', fontsize=12)
    ax3.set_ylabel('Effect on y', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: ALE for test data with true effect
    ax4 = fig.add_subplot(grid[1, 1])
    ax4.plot(ale_test_x1['x1'], ale_test_x1['eff'], 'r-', linewidth=2, label='ALE estimate')
    ax4.fill_between(
        ale_test_x1['x1'],
        ale_test_x1['lowerCI_95%'],
        ale_test_x1['upperCI_95%'],
        alpha=0.2,
        color='r',
        label='95% CI'
    )
    
    # Calculate and add true marginal effect for the test range
    x1_grid_test = np.linspace(min(ale_test_x1['x1']), max(ale_test_x1['x1']), 100)
    
    # Calculate effect at average test x2
    true_effect_test_avg_x2 = calculate_true_marginal_effect(x1_grid_test, avg_x2_test, n_features)
    true_effect_test_avg_x2_scaled = true_effect_test_avg_x2 - np.mean(true_effect_test_avg_x2)
    
    # Calculate average effect over all test x2 values
    true_effect_test_all_x2 = calculate_average_marginal_effect(
        x1_grid_test, 
        x2_range=(X_test[:, 1].min(), X_test[:, 1].max()),
        n_x2_points=20,
        n_features=n_features
    )
    true_effect_test_all_x2_scaled = true_effect_test_all_x2 - np.mean(true_effect_test_all_x2)
    
    # Add both true effects to the plot
    ax4.plot(x1_grid_test, true_effect_test_avg_x2_scaled, 'g--', linewidth=1.5, 
             label=f'True effect at mean x2={avg_x2_test:.2f}')
    ax4.plot(x1_grid_test, true_effect_test_all_x2_scaled, 'r--', linewidth=2, 
             label='True effect averaged over all x2')
    
    # Highlight the test data range
    ax4.set_title('ALE for x1 on Test Data', fontsize=14)
    ax4.set_xlabel('x1', fontsize=12)
    ax4.set_ylabel('Effect on y', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Failure of ALE Under Spatial Covariate Shift', fontsize=16)
    
    # Save figure
    output_dir = 'ale_results'
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, 'ale_spatial_covariate_shift.png')
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
    plt.savefig(os.path.join(output_dir, 'spatial_feature_importance.png'), dpi=300, bbox_inches='tight')
    
    # Create a detailed visualization of true marginal effects
    print("Calculating detailed true marginal effects of x1...")
    
    # Create a grid of points
    x1_grid = np.linspace(0, 1, 100)
    
    # Set up figure for visualizing true marginal effects
    plt.figure(figsize=(15, 8))
    
    # Create a new function to show the heatmap of partial derivatives
    def plot_true_effect_heatmap():
        x1_values = np.linspace(0, 1, 50)
        x2_values = np.linspace(0, 1, 50)
        X1, X2 = np.meshgrid(x1_values, x2_values)
        
        # Calculate partial derivative at each point
        Z = np.zeros_like(X1)
        for i in range(len(x2_values)):
            for j in range(len(x1_values)):
                # Calculate finite difference approximation of partial derivative
                x1, x2 = x1_values[j], x2_values[i]
                delta = 0.001
                
                # Create feature vectors
                X_base = np.zeros(n_features)
                X_base[0], X_base[1] = x1, x2
                
                X_delta = X_base.copy()
                X_delta[0] += delta
                
                # Calculate partial derivative
                y_base = spatial_function(X_base[None, :])
                y_delta = spatial_function(X_delta[None, :])
                Z[i, j] = (y_delta - y_base) / delta
        
        plt.figure(figsize=(10, 8))
        plt.imshow(Z, extent=[0, 1, 0, 1], origin='lower', aspect='auto', cmap='viridis')
        plt.colorbar(label='∂f/∂x1')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Heatmap of True Marginal Effect of x1 (∂f/∂x1)')
        
        # Add rectangles for training and test regions
        training_rect = plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, edgecolor='blue', 
                                    linestyle='--', linewidth=2, label='Training region')
        test_rect = plt.Rectangle((0.6, 0.6), 0.2, 0.2, fill=False, edgecolor='red',
                                linestyle='--', linewidth=2, label='Test region')
        plt.gca().add_patch(training_rect)
        plt.gca().add_patch(test_rect)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'true_effect_heatmap.png'), dpi=300, bbox_inches='tight')
    
    # Calculate and plot true marginal effect at different fixed x2 values
    plt.figure(figsize=(12, 6))
    
    # Include average values from training and test, plus some additional points
    x2_values = [0.2, 0.4, 0.6, 0.8, avg_x2_train, avg_x2_test]
    
    # Also calculate the average over all x2 values
    avg_effect = calculate_average_marginal_effect(
        x1_grid, 
        x2_range=(0.1, 0.9),
        n_x2_points=20,
        n_features=n_features
    )
    
    # Plot individual effects for specific x2 values
    for x2 in x2_values:
        label = f'x2={x2:.2f}'
        if np.isclose(x2, avg_x2_train):
            label += ' (train mean)'
        elif np.isclose(x2, avg_x2_test):
            label += ' (test mean)'
            
        true_effect = calculate_true_marginal_effect(x1_grid, x2, n_features)
        plt.plot(x1_grid, true_effect, '--', alpha=0.7, linewidth=1.5, label=label)
    
    # Plot average effect with thicker line
    plt.plot(x1_grid, avg_effect, 'k-', linewidth=3, label='Average over all x2 values')
    
    # Mark training and test regions
    plt.axvspan(0.1, 0.9, alpha=0.1, color='blue', label='Training x1 range')
    plt.axvspan(0.6, 0.8, alpha=0.1, color='red', label='Test x1 range')
    
    plt.xlabel('x1')
    plt.ylabel('∂f/∂x1')
    plt.title('True Marginal Effect of x1 at Different x2 Values')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'true_marginal_effects.png'), dpi=300, bbox_inches='tight')
    
    # Create the heatmap of marginal effects
    plot_true_effect_heatmap()
    
    # Print key findings
    print("\nKey Findings:")
    print("1. The true effect of x1 depends on x2 due to interactions in the spatial function")
    print("2. ALE can still fail even when test data is contained within the training region")
    print("3. This happens because ALE averages over the distribution of the other features (x2)")
    print("4. When the test distribution is different, the average effect is calculated differently")
    print("5. The true effect averaged over all x2 values provides a more complete picture")


if __name__ == "__main__":
    main() 