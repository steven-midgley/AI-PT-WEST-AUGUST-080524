import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class RMSELinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
    
    def _rmse_loss(self, params, X, y):
        """
        Calculate RMSE loss for optimization
        params[0] is intercept, params[1:] are coefficients
        """
        y_pred = self._predict_with_params(X, params)
        return np.sqrt(np.mean((y - y_pred) ** 2))
    
    def _predict_with_params(self, X, params):
        """Helper method to make predictions during optimization"""
        return X @ params[1:] + params[0]
    
    def fit(self, X, y):
        """
        Fit the model by directly minimizing RMSE
        """
        # Initial guess using normal equations (standard OLS solution)
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        initial_params = np.linalg.pinv(X_with_bias) @ y
        
        # Optimize to minimize RMSE
        result = minimize(
            self._rmse_loss,
            initial_params,
            args=(X, y),
            method='Nelder-Mead'  # You can try other methods like 'BFGS', 'Powell', etc.
        )
        
        # Store the optimized parameters
        self.intercept_ = result.x[0]
        self.coef_ = result.x[1:]
        return self
    
    def predict(self, X):
        """Make predictions with the fitted model"""
        return self._predict_with_params(X, np.r_[self.intercept_, self.coef_])

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.ravel() + 1 + np.random.randn(100) * 0.5

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate the RMSE-optimized model
rmse_model = RMSELinearRegression()
rmse_model.fit(X_train, y_train)

# Make predictions
y_train_pred = rmse_model.predict(X_train)
y_test_pred = rmse_model.predict(X_test)

# Calculate final RMSE
train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))

# Print results
print(f"Model Coefficients: {rmse_model.coef_[0]:.4f}")
print(f"Model Intercept: {rmse_model.intercept_:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing RMSE: {test_rmse:.4f}")