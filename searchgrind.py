import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Reshape the data to include the look-back window dimension
def reshape_data(X, Y, look_back):
    X_list, Y_list = [], []
    for i in range(len(X) - look_back):
        X_list.append(X.iloc[i:(i + look_back)].values)
        Y_list.append(Y.iloc[i + look_back].values)
    return np.array(X_list), np.array(Y_list)

# Normalize the features and target
def normalize_data(X, Y):
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()

    X_normalized = X_scaler.fit_transform(X)
    Y_normalized = Y_scaler.fit_transform(Y.reshape(-1, 1))

    return X_normalized, Y_normalized, X_scaler, Y_scaler

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

# Normalize the data
X_normalized, Y_normalized, X_scaler, Y_scaler = normalize_data(X, Y)

# Prepare the data with a default look-back window size
look_back = 1
X_reshaped, Y_reshaped = reshape_data(X_normalized, Y_normalized, look_back)

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_reshaped, Y_reshaped, test_size=0.2, random_state=42)

# Initial hyperparameters
best_params = {
    'units': 50,
    'dense_nodes': 10,
    'activation': 'relu',
    'optimizer': 'adam',
    'dropout_rate': 0.2,
    'num_lstm_layers': 1,
    'num_gru_layers': 0,
    'num_dense_layers': 1,
    'look_back': 1,
    'batch_size': 32,
    'epochs': 10
}

# Function to evaluate the model and return validation loss
def evaluate_model(params):
    model = build_model(**params)
    model.fit(X_train, Y_train, batch_size=params['batch_size'], epochs=params['epochs'], verbose=0)
    loss = model.evaluate(X_val, Y_val, verbose=0)
    return loss

# Greedy dynamic search function
def dynamic_search(best_params, max_iterations=10, tolerance=1e-3):
    best_score = evaluate_model(best_params)
    param_steps = {
        'units': [10, -10],
        'dense_nodes': [5, -5],
        'dropout_rate': [0.05, -0.05],
        'num_lstm_layers': [1, -1],
        'num_gru_layers': [1, -1],
        'num_dense_layers': [1, -1],
        'look_back': [1, -1],
        'batch_size': [16, -16],
        'epochs': [10, -10]
    }
    
    activation_options = ['relu', 'tanh', 'sigmoid']
    optimizer_options = ['adam', 'rmsprop', 'sgd']
    
    for iteration in range(max_iterations):
        improved = False
        
        # Try different activation functions
        for activation in activation_options:
            if activation != best_params['activation']:
                current_params = best_params.copy()
                current_params['activation'] = activation
                score = evaluate_model(current_params)
                if score < best_score - tolerance:
                    best_score = score
                    best_params['activation'] = activation
                    improved = True
                    print(f"Iteration {iteration + 1}: Changed activation to {activation} with score {best_score}")
        
        # Try different optimizers
        for optimizer in optimizer_options:
            if optimizer != best_params['optimizer']:
                current_params = best_params.copy()
                current_params['optimizer'] = optimizer
                score = evaluate_model(current_params)
                if score < best_score - tolerance:
                    best_score = score
                    best_params['optimizer'] = optimizer
                    improved = True
                    print(f"Iteration {iteration + 1}: Changed optimizer to {optimizer} with score {best_score}")
        
        # Try variations in numerical parameters
        for key in param_steps:
            current_value = best_params[key]
            for step in param_steps[key]:
                new_value = current_value + step
                if new_value > 0:  # Ensure the parameters remain positive and valid
                    current_params = best_params.copy()
                    current_params[key] = new_value
                    score = evaluate_model(current_params)
                    if score < best_score - tolerance:  # Improvement found
                        best_score = score
                        best_params[key] = new_value
                        improved = True
                        print(f"Iteration {iteration + 1}: Improved {key} to {new_value} with score {best_score}")
        
        if not improved:  # Stop if no improvement found
            print(f"No further improvements after {iteration + 1} iterations.")
            break
    
    return best_params, best_score

# Perform the dynamic search
best_params, best_score = dynamic_search(best_params)

print(f"Best parameters found: {best_params}")
print(f"Best validation score: {best_score}")

# Optionally save the best model
# best_model = build_model(**best_params)
# best_model.save('best_model.keras')













#####################################################################################################################################################################################################################################################################################################################################














import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# Reshape the data to include the look-back window dimension
def reshape_data(X, Y, look_back):
    X_list, Y_list = [], []
    for i in range(len(X) - look_back):
        X_list.append(X.iloc[i:(i + look_back)].values)
        Y_list.append(Y.iloc[i + look_back].values)
    return np.array(X_list), np.array(Y_list)

# Normalize the features and target
def normalize_data(X, Y):
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()

    X_normalized = X_scaler.fit_transform(X)
    Y_normalized = Y_scaler.fit_transform(Y.reshape(-1, 1))

    return X_normalized, Y_normalized, X_scaler, Y_scaler

# Model builder function
def build_model(units, dense_nodes, activation, optimizer, dropout_rate, num_lstm_layers, num_gru_layers, num_dense_layers, look_back):
    model = Sequential()
    
    # Add LSTM layers
    for i in range(num_lstm_layers):
        if i == 0:  # First layer needs to specify input shape
            model.add(LSTM(units, input_shape=(look_back, X_train.shape[2]), return_sequences=True))
        else:
            model.add(LSTM(units, return_sequences=(i < num_lstm_layers - 1)))
        model.add(Dropout(dropout_rate))
    
    # Add GRU layers
    for i in range(num_gru_layers):
        if num_lstm_layers == 0 and i == 0:  # If no LSTM layers, first GRU layer specifies input shape
            model.add(GRU(units, input_shape=(look_back, X_train.shape[2]), return_sequences=True))
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

# Normalize the data
X_normalized, Y_normalized, X_scaler, Y_scaler = normalize_data(X, Y)

# Prepare the data with a default look-back window size
look_back = 1
X_reshaped, Y_reshaped = reshape_data(X_normalized, Y_normalized, look_back)

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_reshaped, Y_reshaped, test_size=0.2, random_state=42)

# Define the hyperparameter search space
space = [
    Integer(10, 100, name='units'),
    Integer(5, 50, name='dense_nodes'),
    Real(0.0, 0.5, name='dropout_rate'),
    Integer(1, 3, name='num_lstm_layers'),
    Integer(0, 3, name='num_gru_layers'),
    Integer(1, 3, name='num_dense_layers'),
    Categorical(['relu', 'tanh', 'sigmoid'], name='activation'),
    Categorical(['adam', 'rmsprop', 'sgd'], name='optimizer'),
    Integer(1, 5, name='look_back'),
    Integer(16, 64, name='batch_size'),
    Integer(10, 50, name='epochs')
]

@use_named_args(space)
def objective(**params):
    model = build_model(**params)
    model.fit(X_train, Y_train, batch_size=params['batch_size'], epochs=params['epochs'], verbose=0)
    loss = model.evaluate(X_val, Y_val, verbose=0)
    return loss

# Run Bayesian Optimization
result = gp_minimize(objective, space, n_calls=50, random_state=42)

# Best hyperparameters found
best_params = dict(zip(
    ['units', 'dense_nodes', 'dropout_rate', 'num_lstm_layers', 'num_gru_layers', 'num_dense_layers', 'activation', 'optimizer', 'look_back', 'batch_size', 'epochs'],
    result.x
))

print(f"Best hyperparameters: {best_params}")
print(f"Best validation loss: {result.fun}")

# Optionally save the best model
# best_model = build_model(**best_params)
# best_model.fit(X_train, Y_train, batch_size=best_params['batch_size'], epochs=best_params['epochs'], verbose=0)
# best_model.save('best_model.keras')
