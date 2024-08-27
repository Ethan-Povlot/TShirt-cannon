import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Attention, Concatenate, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.interpolate import interp1d
from statsmodels.tsa.seasonal import seasonal_decompose
import tsaug
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

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
def create_sequences(X, y, lookback):
    sequences_X, sequences_y = [], []
    for i in range(len(X) - lookback):
        seq_X = X[i:i + lookback]
        seq_y = y[i + lookback]
        sequences_X.append(seq_X)
        sequences_y.append(seq_y)
    return np.array(sequences_X), np.array(sequences_y)

X_sequences, y_sequences = create_sequences(augmented_data, y_target_scaled, lookback)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, shuffle=False)

# Define the model creation function
def create_model(
        units=50, dense_units=50, dropout_rate=0.2, learning_rate=0.001,
        num_layers=1, num_dense_layers=1, use_gru=False, use_bidirectional=False,
        use_attention=False):
    
    model = Sequential()
    
    # Add LSTM or GRU layers
    for _ in range(num_layers):
        if use_bidirectional:
            if use_gru:
                model.add(Bidirectional(GRU(units, activation='relu', return_sequences=True if _ < num_layers - 1 else False, input_shape=(lookback, X_features_scaled.shape[1]))))
            else:
                model.add(Bidirectional(LSTM(units, activation='relu', return_sequences=True if _ < num_layers - 1 else False, input_shape=(lookback, X_features_scaled.shape[1]))))
        else:
            if use_gru:
                model.add(GRU(units, activation='relu', return_sequences=True if _ < num_layers - 1 else False, input_shape=(lookback, X_features_scaled.shape[1])))
            else:
                model.add(LSTM(units, activation='relu', return_sequences=True if _ < num_layers - 1 else False, input_shape=(lookback, X_features_scaled.shape[1])))
        model.add(Dropout(dropout_rate))
    
    # Optionally add Attention layer
    if use_attention:
        model.add(Attention())
    
    # Flatten the output if Attention is used
    if use_attention:
        model.add(Flatten())
    
    # Add Dense layers
    for _ in range(num_dense_layers):
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Define hyperparameter space for Randomized Search
def objective(params):
    model = create_model(**params)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)
    
    model.fit(X_train, y_train, epochs=50, batch_size=params['batch_size'], validation_split=0.1, callbacks=[early_stopping, model_checkpoint], verbose=0)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {'loss': mse, 'status': STATUS_OK}

space = {
    'units': hp.choice('units', [50, 100]),
    'dense_units': hp.choice('dense_units', [50, 100]),
    'dropout_rate': hp.uniform('dropout_rate', 0, 0.5),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-1)),
    'batch_size': hp.choice('batch_size', [16, 32]),
    'num_layers': hp.choice('num_layers', [1, 2]),
    'num_dense_layers': hp.choice('num_dense_layers', [1, 2]),
    'use_gru': hp.choice('use_gru', [True, False]),
    'use_bidirectional': hp.choice('use_bidirectional', [True, False]),
    'use_attention': hp.choice('use_attention', [True, False])
}

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

# Load and evaluate the best model
best_params = best
best_model = create_model(**best_params)
best_model.load_weights('best_model.h5')
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Best Model MSE: {mse}")





###################################################################################################################################################################################################################################







import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Attention, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import tensorflow as tf
from scipy.interpolate import interp1d
from statsmodels.tsa.seasonal import seasonal_decompose

# Data Augmentation Functions
def create_windows(data, window_size):
    return np.array([data[i:i + window_size] for i in range(len(data) - window_size + 1)])

def time_warp(data, new_length):
    x = np.arange(len(data))
    x_new = np.linspace(0, len(data) - 1, new_length)
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
    return trend + seasonal + residual * 0.5

# Load and Prepare Data
df = pd.DataFrame({
    'X': np.random.randn(1000),
    'y': np.random.randn(1000)
})

lookback = 30

# Create lagged features
for i in range(1, lookback + 1):
    df[f'y_lag_{i}'] = df['y'].shift(i)

df = df.dropna()

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

def create_sequences(X, y, lookback):
    sequences_X, sequences_y = [], []
    for i in range(len(X) - lookback):
        seq_X = X[i:i + lookback]
        seq_y = y[i + lookback]
        sequences_X.append(seq_X)
        sequences_y.append(seq_y)
    return np.array(sequences_X), np.array(sequences_y)

X_sequences, y_sequences = create_sequences(augmented_data, y_target_scaled, lookback)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, shuffle=False)

# Define the model creation function
def create_model(
        units=50, dense_units=50, dropout_rate=0.2, learning_rate=0.001,
        num_layers=1, num_dense_layers=1, use_gru=False, use_bidirectional=False,
        use_attention=False, use_batch_norm=False):
    
    model = Sequential()
    input_shape = (lookback, X_features_scaled.shape[1])

    for _ in range(num_layers):
        if use_bidirectional:
            layer = GRU(units, activation='relu', return_sequences=True, input_shape=input_shape) if use_gru else LSTM(units, activation='relu', return_sequences=True, input_shape=input_shape)
            model.add(Bidirectional(layer))
        else:
            layer = GRU(units, activation='relu', return_sequences=True, input_shape=input_shape) if use_gru else LSTM(units, activation='relu', return_sequences=True, input_shape=input_shape)
            model.add(layer)
        if use_batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    if use_attention:
        model.add(Attention())
        model.add(Flatten())
    
    for _ in range(num_dense_layers):
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Hyperparameter optimization
def objective(params):
    model = create_model(**params)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

    model.fit(X_train, y_train, epochs=50, batch_size=params['batch_size'], validation_split=0.1, callbacks=[early_stopping, reduce_lr, model_checkpoint], verbose=0)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {'loss': mse, 'status': STATUS_OK}

space = {
    'units': hp.choice('units', [50, 100, 150]),
    'dense_units': hp.choice('dense_units', [50, 100, 150]),
    'dropout_rate': hp.uniform('dropout_rate', 0, 0.5),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-1)),
    'batch_size': hp.choice('batch_size', [16, 32, 64]),
    'num_layers': hp.choice('num_layers', [1, 2, 3]),
    'num_dense_layers': hp.choice('num_dense_layers', [1, 2, 3]),
    'use_gru': hp.choice('use_gru', [True, False]),
    'use_bidirectional': hp.choice('use_bidirectional', [True, False]),
    'use_attention': hp.choice('use_attention', [True, False]),
    'use_batch_norm': hp.choice('use_batch_norm', [True, False])
}

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

# Load and evaluate the best model
best_params = best
best_model = create_model(**best_params)
best_model.load_weights('best_model.h5')
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Best Model MSE: {mse}")




##############################################################################################################################################################

# option for classifier search 

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional, Attention, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import tensorflow as tf
from scipy.interpolate import interp1d
from statsmodels.tsa.seasonal import seasonal_decompose

# Data Augmentation Functions
def create_windows(data, window_size):
    return np.array([data[i:i + window_size] for i in range(len(data) - window_size + 1)])

def time_warp(data, new_length):
    x = np.arange(len(data))
    x_new = np.linspace(0, len(data) - 1, new_length)
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
    return trend + seasonal + residual * 0.5

# Load and Prepare Data
df = pd.DataFrame({
    'X': np.random.randn(1000),
    'y': np.random.randint(0, 3, 1000)  # Randomly assigning one of three classes
})

lookback = 30

# Create lagged features
for i in range(1, lookback + 1):
    df[f'y_lag_{i}'] = df['y'].shift(i)

df = df.dropna()

X_features = df[[f'y_lag_{i}' for i in range(1, lookback + 1)] + ['X']].values
y_target = df['y'].values

# Standardize the features
scaler_X = StandardScaler()
X_features_scaled = scaler_X.fit_transform(X_features)

# One-hot encode the target
onehot_encoder = OneHotEncoder(sparse=False)
y_target_encoded = onehot_encoder.fit_transform(y_target.reshape(-1, 1))

# Data Augmentation
windows = create_windows(df['y'].values, lookback)
augmented_data = np.concatenate([
    windows,
    time_warp(df['y'].values, len(df['y']) * 2).reshape(-1, 1),
    add_noise(df['y'].values),
    jitter(df['y'].values),
    seasonal_decomposition_augmentation(df['y'].values).reshape(-1, 1)
], axis=0)

def create_sequences(X, y, lookback):
    sequences_X, sequences_y = [], []
    for i in range(len(X) - lookback):
        seq_X = X[i:i + lookback]
        seq_y = y[i + lookback]
        sequences_X.append(seq_X)
        sequences_y.append(seq_y)
    return np.array(sequences_X), np.array(sequences_y)

X_sequences, y_sequences = create_sequences(augmented_data, y_target_encoded, lookback)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_sequences, test_size=0.2, shuffle=False)

# Define the model creation function
def create_model(
        units=50, dense_units=50, dropout_rate=0.2, learning_rate=0.001,
        num_layers=1, num_dense_layers=1, use_gru=False, use_bidirectional=False,
        use_attention=False, use_batch_norm=False):
    
    model = Sequential()
    input_shape = (lookback, X_features_scaled.shape[1])

    for _ in range(num_layers):
        if use_bidirectional:
            layer = GRU(units, activation='relu', return_sequences=True, input_shape=input_shape) if use_gru else LSTM(units, activation='relu', return_sequences=True, input_shape=input_shape)
            model.add(Bidirectional(layer))
        else:
            layer = GRU(units, activation='relu', return_sequences=True, input_shape=input_shape) if use_gru else LSTM(units, activation='relu', return_sequences=True, input_shape=input_shape)
            model.add(layer)
        if use_batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    if use_attention:
        model.add(Attention())
        model.add(Flatten())
    
    for _ in range(num_dense_layers):
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(3, activation='softmax'))  # Output layer for 3 classes
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Hyperparameter optimization
def objective(params):
    model = create_model(**params)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

    model.fit(X_train, y_train, epochs=50, batch_size=params['batch_size'], validation_split=0.1, callbacks=[early_stopping, reduce_lr, model_checkpoint], verbose=0)
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_classes, y_pred_classes)
    return {'loss': 1 - accuracy, 'status': STATUS_OK}

space = {
    'units': hp.choice('units', [50, 100, 150]),
    'dense_units': hp.choice('dense_units', [50, 100, 150]),
    'dropout_rate': hp.uniform('dropout_rate', 0, 0.5),
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-1)),
    'batch_size': hp.choice('batch_size', [16, 32, 64]),
    'num_layers': hp.choice('num_layers', [1, 2, 3]),
    'num_dense_layers': hp.choice('num_dense_layers', [1, 2, 3]),
    'use_gru': hp.choice('use_gru', [True, False]),
    'use_bidirectional': hp.choice('use_bidirectional', [True, False]),
    'use_attention': hp.choice('use_attention', [True, False]),
    'use_batch_norm': hp.choice('use_batch_norm', [True, False])
}

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

# Load and evaluate the best model
best_params = best
best_model = create_model(**best_params)
best_model.load_weights('best_model.h5')
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_classes, y_pred_classes)
print(f"Best Model Accuracy: {accuracy}")

print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes))
