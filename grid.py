import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Reshape the data to include the look-back window dimension
def reshape_data(X, Y, look_back):
    X_list, Y_list = [], []
    for i in range(len(X) - look_back):
        X_list.append(X.iloc[i:(i + look_back)].values)
        Y_list.append(Y.iloc[i + look_back].values)
    return np.array(X_list), np.array(Y_list)

# Model builder function
def build_model(units=50, dense_nodes=10, activation='relu', optimizer='adam', 
                dropout_rate=0.2, num_lstm_layers=1, num_gru_layers=0, 
                num_dense_layers=1, look_back=1):
    
    model = Sequential()
    
    # Add LSTM layers
    for i in range(num_lstm_layers):
        if i == 0:  # First layer needs to specify input shape
            model.add(LSTM(units, input_shape=(look_back, X.shape[1]), return_sequences=True))
        else:
            model.add(LSTM(units, return_sequences=(i < num_lstm_layers - 1)))
        model.add(Dropout(dropout_rate))
    
    # Add GRU layers
    for i in range(num_gru_layers):
        if num_lstm_layers == 0 and i == 0:  # If no LSTM layers, first GRU layer specifies input shape
            model.add(GRU(units, input_shape=(look_back, X.shape[1]), return_sequences=True))
        else:
            model.add(GRU(units, return_sequences=(i < num_gru_layers - 1)))
        model.add(Dropout(dropout_rate))
    
    # Add dense layers
    for _ in range(num_dense_layers):
        model.add(Dense(dense_nodes, activation=activation))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Prepare the data with a default look-back window size for GridSearchCV
X_reshaped, Y_reshaped = reshape_data(X, Y, look_back=1)

# Wrap the model with KerasClassifier
model = KerasClassifier(
    model=build_model, 
    units=50, 
    dense_nodes=10, 
    activation='relu', 
    optimizer='adam',
    dropout_rate=0.2,
    num_lstm_layers=1,
    num_gru_layers=0,
    num_dense_layers=1,
    look_back=1,
    batch_size=None,  # Allows grid search to override
    epochs=10,  # Default epochs value; can be overridden by grid search
    verbose=0
)

# Grid search parameters including batch size
param_grid = {
    'units': [50, 100],  # Number of LSTM/GRU units
    'dense_nodes': [10, 20],  # Number of dense nodes
    'activation': ['relu', 'tanh'],  # Activation function for dense layer
    'optimizer': ['adam', 'rmsprop'],  # Optimizer
    'dropout_rate': [0.2, 0.3],  # Dropout rate
    'num_lstm_layers': [1, 2],  # Number of LSTM layers
    'num_gru_layers': [0, 1],  # Number of GRU layers
    'num_dense_layers': [1, 2],  # Number of dense layers
    'look_back': [1, 3, 5],  # Look-back window size
    'batch_size': [16, 32, 64],  # Batch sizes
    'epochs': [10, 20]  # Number of epochs
}

# Grid search with cross-validation
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_result = grid.fit(X_reshaped, Y_reshaped)

# Display the best parameters and score
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

# Optionally save the best model
# best_model = grid_result.best_estimator_.model_
# best_model.save('best_model.keras')
