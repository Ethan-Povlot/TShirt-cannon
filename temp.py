# Updated model builder function with different units per layer for LSTM, GRU, and Dense layers and list for dropout rates
def build_model(lstm_units_list=None, gru_units_list=None, dense_node_list=None, 
                dropout_rate_list=None, activation='relu', optimizer='adam', input_shape=None):
    
    model = Sequential()
    
    # Add LSTM layers with different units and dropout rates for each layer
    if lstm_units_list is not None:
        for i, units in enumerate(lstm_units_list):
            if i == 0:  # First LSTM layer needs to specify input shape
                model.add(LSTM(units, input_shape=input_shape, return_sequences=(i < len(lstm_units_list) - 1)))
            else:
                model.add(LSTM(units, return_sequences=(i < len(lstm_units_list) - 1)))
            if dropout_rate_list:
                model.add(Dropout(dropout_rate_list[i]))
    
    # Add GRU layers with different units and dropout rates for each layer
    if gru_units_list is not None:
        for i, units in enumerate(gru_units_list):
            if lstm_units_list is None and i == 0:  # If no LSTM layers, the first GRU layer needs input shape
                model.add(GRU(units, input_shape=input_shape, return_sequences=(i < len(gru_units_list) - 1)))
            else:
                model.add(GRU(units, return_sequences=(i < len(gru_units_list) - 1)))
            if dropout_rate_list:
                model.add(Dropout(dropout_rate_list[len(lstm_units_list) + i]))
    
    # Add dense layers with different nodes and dropout rates for each layer
    if dense_node_list is not None:
        for i, nodes in enumerate(dense_node_list):
            model.add(Dense(nodes, activation=activation))
            if dropout_rate_list:
                model.add(Dropout(dropout_rate_list[len(lstm_units_list) + len(gru_units_list) + i]))
    
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer=optimizer, loss='mse')
    return model

# Simulated Annealing Dynamic Search function
def simulated_annealing_search(initial_params, X_train, Y_train, X_val, Y_val, max_iterations=10, initial_temperature=1.0, cooling_rate=0.9, tolerance=1e-3):
    current_params = initial_params.copy()
    best_params = initial_params.copy()
    best_score = evaluate_model(best_params, X_train, Y_train, X_val, Y_val)
    
    temperature = initial_temperature
    
    for iteration in range(max_iterations):
        improved = False
        
        # Scale the step sizes based on the current temperature
        param_steps = {
            'lstm_units_list': [int(temperature * 10), int(-temperature * 10)],  # Change by 10 units
            'gru_units_list': [int(temperature * 10), int(-temperature * 10)],   # Change by 10 units
            'dense_node_list': [int(temperature * 5), int(-temperature * 5)],    # Change by 5 units
            'dropout_rate_list': [temperature * 0.05, -temperature * 0.05],      # Change dropout rate by 0.05
            'look_back': [int(temperature * 1), int(-temperature * 1)],          # Change look_back window by 1
            'batch_size': [int(temperature * 16), int(-temperature * 16)],       # Change batch size
            'epochs': [int(temperature * 10), int(-temperature * 10)]            # Change number of epochs
        }
        
        activation_options = ['relu', 'tanh', 'sigmoid']
        optimizer_options = ['adam', 'rmsprop', 'sgd']
        
        # Try different activation functions
        for activation in activation_options:
            if activation != current_params['activation']:
                new_params = current_params.copy()
                new_params['activation'] = activation
                score = evaluate_model(new_params, X_train, Y_train, X_val, Y_val)
                if score < best_score - tolerance:
                    best_score = score
                    best_params = new_params
                    improved = True
                    print(f"Iteration {iteration + 1}: Changed activation to {activation} with score {best_score}")
        
        # Try different optimizers
        for optimizer in optimizer_options:
            if optimizer != current_params['optimizer']:
                new_params = current_params.copy()
                new_params['optimizer'] = optimizer
                score = evaluate_model(new_params, X_train, Y_train, X_val, Y_val)
                if score < best_score - tolerance:
                    best_score = score
                    best_params = new_params
                    improved = True
                    print(f"Iteration {iteration + 1}: Changed optimizer to {optimizer} with score {best_score}")
        
        # Try variations in numerical parameters
        for key in param_steps:
            current_value = current_params[key]
            if isinstance(current_value, list):  # Handle lists like lstm_units_list, gru_units_list, etc.
                for i in range(len(current_value)):
                    for step in param_steps[key]:
                        new_value = current_value[i] + step
                        if new_value > 0:  # Ensure the parameters remain positive and valid
                            new_params = current_params.copy()
                            new_params[key][i] = new_value
                            score = evaluate_model(new_params, X_train, Y_train, X_val, Y_val)
                            if score < best_score - tolerance:  # Improvement found
                                best_score = score
                                best_params = new_params
                                improved = True
                                print(f"Iteration {iteration + 1}: Improved {key} to {new_value} with score {best_score}")
            else:  # Handle scalar parameters like look_back, batch_size, etc.
                for step in param_steps[key]:
                    new_value = current_value + step
                    if new_value > 0:  # Ensure the parameters remain positive and valid
                        new_params = current_params.copy()
                        new_params[key] = new_value
                        score = evaluate_model(new_params, X_train, Y_train, X_val, Y_val)
                        if score < best_score - tolerance:  # Improvement found
                            best_score = score
                            best_params = new_params
                            improved = True
                            print(f"Iteration {iteration + 1}: Improved {key} to {new_value} with score {best_score}")
        
        if not improved:  # Stop if no improvement found
            print(f"No further improvements after {iteration + 1} iterations.")
            break
        
        # Reduce the temperature
        temperature *= cooling_rate
    
    return best_params, best_score

# Initial hyperparameters
initial_best_params = {
    'lstm_units_list': [50],
    'gru_units_list': [],
    'dense_node_list': [10],
    'dropout_rate_list': [0.2],  # One value for each layer, should match layer counts
    'activation': 'relu',
    'optimizer': 'adam',
    'look_back': 1,
    'batch_size': 32,
    'epochs': 10
}

# Assuming you have 10 datasets: X1, X2, ..., X10 and corresponding Y1, Y2, ..., Y10
datasets = [(X1, Y1), (X2, Y2), ..., (X10, Y10)]  # Replace with your actual data

# Step 1: Normalize and combine all datasets
X_combined = []
Y_combined = []
scalers = []

for X, Y in datasets:
    X_normalized, Y_normalized, X_scaler, Y_scaler = normalize_data(X, Y)
    X_combined.append(X_normalized)
    Y_combined.append(Y_normalized)
    scalers.append((X_scaler, Y_scaler))

X_combined = np.vstack(X_combined)
Y_combined = np.vstack(Y_combined)

# Reshape combined data
look_back = initial_best_params['look_back']
X_reshaped, Y_reshaped = reshape_data(pd.DataFrame(X_combined), pd.DataFrame(Y_combined), look_back)

# Split the combined data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_reshaped, Y_reshaped, test_size=0.2, random_state=42)

# Step 2: Perform multiple random instantiations with simulated annealing
num_instantiations = 5  # Number of random instantiations
best_global_params = None
best_global_score = float('inf')

for i in range(num_instantiations):
    print(f"Random instantiation {i+1}")
    
    # Randomly initialize the model parameters within a reasonable range
    random_params = {
        'lstm_units_list': [np.random.randint(20, 100) for _ in range(np.random.randint(1, 3))],
        'gru_units_list': [np.random.randint(20, 100) for _ in range(np.random.randint(0, 2))],
        'dense_node_list': [np.random.randint(5, 50) for _ in range(np.random.randint(1, 3))],
        'dropout_rate_list': [np.random.uniform(0.1, 0.5) for _ in range(len(random_params['lstm_units_list']) + len(random_params['gru_units_list']) + len(random_params['dense_node_list']))],
        'activation': np.random.choice(['relu', 'tanh', 'sigmoid']),
        'optimizer': np.random.choice(['adam', 'rmsprop', 'sgd']),
        'look_back': np.random.randint(1, 5),
        'batch_size': np.random.choice([16, 32, 64]),
        'epochs': np.random.choice([10, 20, 30])
    }
    
    # Perform simulated annealing search
    best_params, best_score = simulated_annealing_search(random_params, X_train, Y_train, X_val, Y_val)
    
    if best_score < best_global_score:
        best_global_score = best_score
        best_global_params = best_params
        print(f"New best global parameters found: {best_global_params} with score {best_global_score}")

print(f"Best global parameters after all instantiations: {best_global_params}")
print(f"Best validation score from combined dataset: {best_global_score}")

# Build and train the best model on the combined dataset
base_model = build_model(**best_global_params, input_shape=(X_train.shape[1], X_train.shape[2]))
base_model.fit(X_train, Y_train, batch_size=best_global_params['batch_size'], epochs=best_global_params['epochs'], verbose=1)

# Step 3: Fine-tune on individual datasets
for i, (X, Y) in enumerate(datasets):
    print(f"Fine-tuning on dataset {i+1}")
    
    # Normalize the dataset (reuse scalers from Step 1)
    X_normalized, Y_normalized = scalers[i][0].transform(X), scalers[i][1].transform(Y.reshape(-1, 1))
    
    # Reshape data
    X_reshaped, Y_reshaped = reshape_data(pd.DataFrame(X_normalized), pd.DataFrame(Y_normalized), best_global_params['look_back'])
    
    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X_reshaped, Y_reshaped, test_size=0.2, random_state=42)
    
    # Fine-tune the pre-trained model with the specific dataset
    fine_tuned_model = tf.keras.models.clone_model(base_model)
    fine_tuned_model.set_weights(base_model.get_weights())  # Transfer the weights from the base model
    fine_tuned_model.compile(optimizer=best_global_params['optimizer'], loss='mse')  # Re-compile the model
    fine_tuned_model.fit(X_train, Y_train, batch_size=best_global_params['batch_size'], epochs=best_global_params['epochs'], verbose=1)
    
    # Save the fine-tuned model for this dataset
    model_name = f'fine_tuned_model_dataset_{i+1}.keras'
    fine_tuned_model.save(model_name)
    print(f"Saved fine-tuned model for dataset {i+1} as {model_name}')
