"""
Assignment 6 Part 2: House Price Prediction (Multivariable Regression)

This assignment predicts house prices using MULTIPLE features.
Complete all the functions below following the in-class car price example.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def load_and_explore_data(filename):
    """
    Load the house price data and explore it
    
    Args:
        filename: name of the CSV file to load
    
    Returns:
        pandas DataFrame containing the data
    """
    # TODO: Load the CSV file using pandas
    data = pd.read_csv(filename)
    # TODO: Print the first 5 rows
    print(f"=== Car Price Data ===")
    print(f"\nFirst Five Rows:")
    # TODO: Print the shape of the dataset
    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")
    # TODO: Print basic statistics for ALL columns
    print(f"\nBasic statistics:")
    print(data.describe())
    # TODO: Print the column names
    print(f"\nColumn names: {list(data.columns)}")
    # TODO: Return the dataframe
    return data


def visualize_features(data):
    """
    Create 4 scatter plots (one for each feature vs Price)
    
    Args:
        data: pandas DataFrame with features and Price
    """
    # TODO: Create a figure with 2x2 subplots, size (12, 10)
    fig, axes = plt.subplots(2, 2, figsize=(12,10))
    # TODO: Add a main title: 'House Features vs Price'
    fig.suptitle('House Features vs Price', fontsize=16, fontweight='bold')
    # TODO: Plot 1 (top left): SquareFeet vs Price
    #       - scatter plot, color='blue', alpha=0.6
    #       - labels and title
    #       - grid
    axes[0, 0].scatter(data['SquareFeet'], data['Price'], color='mediumturquoise', alpha=0.6)
    axes[0, 0].set_xlabel('SquareFeet')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].set_title('SquareFeet vs Price', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    # TODO: Plot 2 (top right): Bedrooms vs Price
    #       - scatter plot, color='green', alpha=0.6
    #       - labels and title
    #       - grid
    axes[0, 1].scatter(data['Bedrooms'], data['Price'], color='hotpink', alpha=0.6)
    axes[0, 1].set_xlabel('Bedrooms')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].set_title('Bedrooms vs Price', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    # TODO: Plot 3 (bottom left): Bathrooms vs Price
    #       - scatter plot, color='red', alpha=0.6
    #       - labels and title
    #       - grid
    axes[1, 0].scatter(data['Bathrooms'], data['Price ($)'], color='crimson')
    axes[1, 0].set_xlabel('Bathrooms')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].set_title('Bathrooms vs Price', fontweight='bold')
    axes[1, 0].grid(True, alpha = 0.3)
    # TODO: Plot 4 (bottom right): Age vs Price
    #       - scatter plot, color='orange', alpha=0.6
    #       - labels and title
    #       - grid
    axes[1, 0].scatter(data['Age'], data['Price ($)'], color='orange')
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].set_title('Age vs Price', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    # TODO: Use plt.tight_layout() to make plots fit nicely
    plt.tight_layout()
    # TODO: Save the figure as 'feature_plots.png' with dpi=300
    plt.savefig('feature_plots.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Feature plots saved as 'feature_plots.png'")
    # TODO: Show the plot
    plt.show()


def prepare_features(data):
    """
    Separate features (X) from target (y)
    
    Args:
        data: pandas DataFrame with all columns
    
    Returns:
        X - DataFrame with feature columns
        y - Series with target column
    """
    # TODO: Create a list of feature column names
    feature_columns=['SquareFeet', 'Bedrooms', 'Bathrooms', 'Age']
    # TODO: Create X by selecting those columns from data
    X=data[feature_columns]
    # TODO: Create y by selecting the 'Price' column
    y=data['Price ($)']
    # TODO: Print the shape of X and y
    print(f"\n=== Feature Preparation ===")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    # TODO: Print the feature column names
    print(f"\nFeature columns: {list(X.columns)}")
    # TODO: Return X and y
    return X, y


def split_data(X, y):
    """
    Split data into training and testing sets
    
    Args:
        X: features DataFrame
        y: target Series
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # TODO: Split into train (80%) and test (20%) with random_state=42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    # TODO: Print how many samples are in training and testing sets
    print(f"\n=== Data Split (Matching Unplugged Activity) ===")
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    # TODO: Return X_train, X_test, y_train, y_test
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, feature_names):
    """
    Train a multivariable linear regression model
    
    Args:
        X_train: training features (scaled)
        y_train: training target values
        feature_names: list of feature column names
    
    Returns:
        trained LinearRegression model
    """
    # TODO: Create a LinearRegression model
    model = LinearRegression()
    # TODO: Train the model using fit()
    model.fit(X_train, y_train)
    # TODO: Print the intercept
    print(f"\n=== Model Training Complete ===")
    print(f"Intercept: ${model.intercept_:.2f}")
    # TODO: Print each coefficient with its feature name
    #       Hint: use zip(feature_names, model.coef_)
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.2f}")
    # TODO: Print the full equation in readable format
    equation = f"Price = "
    for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
        if i == 0:
            equation += f"{coef:.2f} × {name}"
        else:
            equation += f" + ({coef:.2f}) × {name}"
    equation += f" + {model.intercept_:.2f}"
    print(equation)
    # TODO: Return the trained model
    return model


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate the model's performance
    
    Args:
        model: trained model
        X_test: testing features (scaled)
        y_test: testing target values
        feature_names: list of feature names
    
    Returns:
        predictions array
    """
    # TODO: Make predictions on X_test
    predictions = model.predict(X_test)
    # TODO: Calculate R² score
    r2 = r2_score(y_test, predictions)
    # TODO: Calculate MSE and RMSE
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    # TODO: Print R² score with interpretation
    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"  → Model explains {r2*100:.2f}% of price variation")
    # TODO: Print RMSE with interpretation
    print(f"\nRoot Mean Squared Error: ${rmse:.2f}")
    print(f"  → On average, predictions are off by ${rmse:.2f}")
    # TODO: Calculate and print feature importance
    #       Hint: Use np.abs(model.coef_) and sort by importance
    #       Show which features matter most
    print(f"\n=== Feature Importance ===")
    feature_importance = list(zip(feature_names, np.abs(model.coef_)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, importance) in enumerate(feature_importance, 1):
        print(f"{i}. {name}: {importance:.2f}")
    # TODO: Return predictions
    return predictions


def compare_predictions(y_test, predictions, num_examples=5):
    """
    Show side-by-side comparison of actual vs predicted prices
    
    Args:
        y_test: actual prices
        predictions: predicted prices
        num_examples: number of examples to show
    """
    # TODO: Print a header row with columns:
    #       Actual Price, Predicted Price, Error, % Error
    print(f"\n=== Prediction Examples ===")
    print(f"{'Actual Price':<15} {'Predicted Price':<18} {'Error':<12} {'% Error'}")
    print("-" * 60)
    # TODO: For the first num_examples:
    #       - Get actual and predicted price
    #       - Calculate error (actual - predicted)
    #       - Calculate percentage error
    #       - Print in a nice formatted table
    for i in range(min(num_examples, len(y_test))):
        actual = y_test.iloc[i]
        predicted = predictions[i]
        error = actual - predicted
        pct_error = (abs(error) / actual) * 100
        
        print(f"${actual:>13.2f}   ${predicted:>13.2f}   ${error:>10.2f}   {pct_error:>6.2f}%")


def make_prediction(model, sqft, bedrooms, bathrooms, age):
    """
    Make a prediction for a specific house
    
    Args:
        model: trained LinearRegression model
        sqft: square footage
        bedrooms: number of bedrooms
        bathrooms: number of bathrooms
        age: age of house in years
    
    Returns:
        predicted price
    """
    # TODO: Create a DataFrame with the house features
    #       columns should be: ['SquareFeet', 'Bedrooms', 'Bathrooms', 'Age']
    
    # TODO: Make a prediction using model.predict()
    
    # TODO: Print the house specs and predicted price nicely formatted
    
    # TODO: Return the predicted price
    pass


if __name__ == "__main__":
    print("=" * 70)
    print("HOUSE PRICE PREDICTION - YOUR ASSIGNMENT")
    print("=" * 70)
    
    # Step 1: Load and explore
    # TODO: Call load_and_explore_data() with 'house_prices.csv'
    data=load_and_explore_data('house_prices.csv')
    # Step 2: Visualize features
    # TODO: Call visualize_features() with the data
    visualize_features(data)
    # Step 3: Prepare features
    # TODO: Call prepare_features() and store X and y
    X, y = prepare_features(data)
    # Step 4: Split data
    # TODO: Call split_data() and store X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Step 5: Train model
    # TODO: Call train_model() with training data and feature names (X.columns)
    model = train_model(X_train, X_test, y_train, y_test)
    # Step 6: Evaluate model
    # TODO: Call evaluate_model() with model, test data, and feature names
    predictions = evaluate_model(model, X_test, y_test, X.columns)
    # Step 7: Compare predictions
    # TODO: Call compare_predictions() showing first 10 examples
    
    # Step 8: Make a new prediction
    # TODO: Call make_prediction() for a house of your choice
    
    print("\n" + "=" * 70)
    print("✓ Assignment complete! Check your saved plots.")
    print("Don't forget to complete a6_part2_writeup.md!")
    print("=" * 70)