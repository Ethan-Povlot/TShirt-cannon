import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from scipy.interpolate import interp1d
from statsmodels.tsa.seasonal import seasonal_decompose
import tsaug

# Data Augmentation Functions

def create_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i + window_size])
    return np.array(windows)

def time_warp(data, new_length):
    original_length = len(data)
    x = np.arange(original_length)
    x_new = np.linspace(0, original_length - 1, new_length)
    interpolator = interp1d(x, data, kind='linear', fill_value='extrapolate')
    return interpolator(x_new)

def add_noise(data, noise_level=0.01):
    noise = np.random.normal(0, noise_level, len(data))
    return data + noise

def jitter(data, jitter_level=0.01):
    jitter = np.random.uniform(-jitter_level, jitter_level, len(data))
    return data + jitter

def seasonal_decomposition_augmentation(data, period=365):
    result = seasonal_decompose(data, model='additive', period=period)
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid
    modified_data = trend + seasonal + residual * 0.5  # Example of modification
    return modified_data

# Load and Prepare Data
df = pd.DataFrame({
    'X': np.random.randn(1000),
    'y': np.random.randn(1000)
})

lookback = 30  # Example lookback period

# Create lagged features
for i in range(1, lookback + 1):
    df[f'y_lag_{i}'] = df['y'].shift(i)

# Drop rows with NaN values
df = df.dropna()

# Features and target
X_features = df[[f'y_lag_{i}' for i in range(1, lookback + 1)] + ['X']].values
y_target = df['y'].values

# Standardize the features
scaler_X = StandardScaler()
X_features_scaled = scaler_X.fit_transform(X_features)

scaler_y = StandardScaler()
y_target_scaled = scaler_y.fit_transform(y_target.reshape(-1, 1)).reshape(-1)

# Data Augmentation
windows = create_windows(df['y'].values, lookback)
augmented_data = np.concatenate([
    windows,
    time_warp(df['y'].values, len(df['y']) * 2).reshape(-1, 1),
    add_noise(df['y'].values),
    jitter(df['y'].values),
    seasonal_decomposition_augmentation(df['y'].values).reshape(-1, 1)
], axis=0)

# Recreate sequences from augmented data
X_sequences, y_sequences = create_sequences(augmented_data, y_target_scaled, lookback)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, shuffle=False)

# Define the LSTM model creation function
def create_model(lstm_units=50, dense_units=50, dropout_rate=0.0, learning_rate=0.001, num_lstm_layers=1, num_dense_layers=1):
    model = Sequential()
    
    # Add LSTM layers
    for _ in range(num_lstm_layers):
        model.add(LSTM(lstm_units, activation='relu', return_sequences=True if _ < num_lstm_layers - 1 else False, input_shape=(lookback, X_features_scaled.shape[1])))
        model.add(Dropout(dropout_rate))
    
    # Add Dense layers
    for _ in range(num_dense_layers):
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Define hyperparameter grid
param_grid = {
    'lstm_units': [50, 100],
    'dense_units': [50, 100],
    'dropout_rate': [0.0, 0.2],
    'learning_rate': [0.001, 0.01],
    'batch_size': [16, 32],
    'epochs': [10, 20],
    'num_lstm_layers': [1, 2],
    'num_dense_layers': [1, 2]
}

# Wrap the model for scikit-learn
model = KerasRegressor(build_fn=create_model, verbose=0)

# Perform Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, y_train)

# Print the best parameters and score
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

# Evaluate the best model
best_model = grid_result.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test Mean Squared Error: {mse}")
