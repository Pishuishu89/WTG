#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io

def run_model(user_input_percentages):
    """
    Runs the MVP prediction model using the user-provided stat importance percentages.
    The function expects a dictionary with keys for each stat (e.g.:
        { 'PTS': 0, 'AST': 60, 'TRB': 30, 'BLK': 10 }).
    These percentages (user_input_percentages) must sum to 100.
    
    Returns a dictionary of matplotlib Figure objects for potential use in a Flask app.
    
    Returned dictionary contains:
        {
            'feature_importance': fig1,
            'top_5_mvp': fig2,
            'error_metrics': fig3,
            'r2_backtest': fig4
        }
    """
    # ---------------------
    # 1. Load and prepare data
    # ---------------------
    stats = pd.read_csv("player_mvp_stats.csv")
    stats = stats.fillna(0)

    predictors = [
        'G', 'GS', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', '2P',
        '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB',
        'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Year', 'W', 'L', 'W/L%',
        'GB', 'PS/G', 'PA/G', 'SRS'
    ]

    # Split data into training and test (2024) sets
    train_df = stats[stats['Year'] < 2024]
    test_df = stats[stats['Year'] == 2024].copy()

    X_train = train_df[predictors].copy()
    X_test = test_df[predictors].copy()
    y_train = train_df['Share']

    # ---------------------
    # 2. Apply Base Weights to the Data
    # ---------------------
    # Define some base weights (these represent the initial, default multipliers)
    base_weights = {
        'PTS': 1.0,
        'AST': 1.5,
        'TRB': 1.2,
        'BLK': 1.0
    }
    for stat, mult in base_weights.items():
        if stat in X_train.columns:
            X_train[stat] *= mult
            X_test[stat] *= mult

    # ---------------------
    # 3. Train the Base Random Forest Model
    # ---------------------
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict base 2024 shares using the base model
    test_df['Predicted_Share'] = model.predict(X_test)

    # ---------------------
    # 4. Validate User Input and Convert to Model Weights
    # ---------------------
    # Check that the user-provided percentages add up to 100.
    if sum(user_input_percentages.values()) != 100:
        raise ValueError("❌ Your user weights must add up to 100.")

    # The user's percentages are scaled relative to the sum of the base weights.
    base_sum = sum(base_weights.values())  # E.g., 1.0 + 1.5 + 1.2 + 1.0 = 4.7
    user_weights = {
        stat: (user_input_percentages[stat] / 100) * base_sum
        for stat in user_input_percentages
    }
    # Now the user_weights contain the adjusted multipliers based on user input.

    # ---------------------
    # 5. Apply User Weights to the Test Data and Re-Predict
    # ---------------------
    X_test_user = test_df[predictors].copy()
    for stat, mult in user_weights.items():
        if stat in X_test_user.columns:
            X_test_user[stat] *= mult
    # Generate new predictions using the user-adjusted weights
    test_df['User_Predicted_Share'] = model.predict(X_test_user)

    # ---------------------
    # 6. Generate Visualizations
    # ---------------------
    figures = {}

    # (A) Feature Importance (Base Model)
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    importances.sort_values(ascending=False).plot(kind='bar', color='#F28C8C', ax=ax1)
    ax1.set_title("Feature Importances in MVP Prediction (Base Model)")
    fig1.tight_layout()
    figures['feature_importance'] = fig1

    # (B) Top 5 Predicted MVPs (Using User-Defined Weights)
    top_5_user = test_df.sort_values(by='User_Predicted_Share', ascending=False).head(5)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.barh(top_5_user['Player'], top_5_user['User_Predicted_Share'], color='#F28C8C')
    ax2.set_xlabel("Predicted Share")
    ax2.set_title("Top 5 Predicted MVPs (User-Defined Weights)")
    ax2.invert_yaxis()  # highest value on top
    fig2.tight_layout()
    figures['top_5_mvp'] = fig2


        # (C) Error Metrics for 2024 (User-Weighted, Training Data)
    X_train_user = train_df[predictors].copy()
    for stat, mult in user_weights.items():
        if stat in X_train_user.columns:
            X_train_user[stat] *= mult
    y_pred_user_train = model.predict(X_train_user)
    y_true_train = train_df['Share']

    mae_user = mean_absolute_error(y_true_train, y_pred_user_train)
    mse_user = mean_squared_error(y_true_train, y_pred_user_train)
    r2_user = r2_score(y_true_train, y_pred_user_train)

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar(['MAE', 'MSE', 'R²'], [mae_user, mse_user, r2_user], color='#F28C8C')
    ax3.set_title("Error Metrics (User-Weighted, Training Data)")
    fig3.tight_layout()
    figures['error_metrics'] = fig3

    # (D) Backtest R² Over Time (User-Defined Weights)
    user_backtest_years = range(2000, 2023)
    user_results = []

    for year in user_backtest_years:
        train_df_bt = stats[stats['Year'] < year]
        test_df_bt = stats[stats['Year'] == year].copy()
        if train_df_bt.empty or test_df_bt.empty:
            continue

        X_train_bt = train_df_bt[predictors].copy()
        y_train_bt = train_df_bt['Share']
        X_test_bt = test_df_bt[predictors].copy()
        y_test_bt = test_df_bt['Share']

        # Apply user weights to the backtest data
        for stat, mult in user_weights.items():
            if stat in X_train_bt.columns:
                X_train_bt[stat] *= mult
            if stat in X_test_bt.columns:
                X_test_bt[stat] *= mult

        user_model = RandomForestRegressor(n_estimators=100, random_state=42)
        user_model.fit(X_train_bt, y_train_bt)
        y_pred_bt = user_model.predict(X_test_bt)

        mae_bt = mean_absolute_error(y_test_bt, y_pred_bt)
        mse_bt = mean_squared_error(y_test_bt, y_pred_bt)
        r2_bt = r2_score(y_test_bt, y_pred_bt)
        user_results.append({'Year': year, 'MAE': mae_bt, 'MSE': mse_bt, 'R2': r2_bt})

    user_results_df = pd.DataFrame(user_results)

    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.plot(user_results_df['Year'], user_results_df['R2'],
             marker='o', linestyle='-', color='#F28C8C', label='R² Score')
    ax4.axhline(0.5, color='gray', linestyle='--', label='0.5 Threshold')
    ax4.set_title("Model R² Over Time (User-Defined Stat Weights)")
    ax4.set_xlabel("Year")
    ax4.set_ylabel("R² Score")
    ax4.legend()
    ax4.grid(True)
    fig4.tight_layout()
    figures['r2_backtest'] = fig4

    # ---------------------
    # 7. Return the Figures
    # ---------------------
    # All figures generated are returned in a dictionary.
    return figures
# -----------------------------------
# OPTIONAL: Standalone test run
# -----------------------------------
if __name__ == "__main__":
    # Example usage with user-defined weights that sum to 100.
    # This dictionary is the user_input_percentages that will be passed to the function.
    example_weights = {'PTS': 0, 'AST': 0, 'TRB': 0, 'BLK': 0} #Change these if saving to laptop & make sure the values add up to 100 or it won't work 

    figs = run_model(example_weights)

    # Save the generated plots to files for local verification.
    figs['feature_importance'].savefig("feature_importance.png")
    figs['top_5_mvp'].savefig("top_5_mvp.png")
    figs['error_metrics'].savefig("error_metrics.png")
    print("All figures saved. Script finished.")
