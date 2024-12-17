# Insurance Cost Prediction Neural Network

# Cell 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cell 2: Load and Inspect Data
# Assuming the data is saved as 'insurance_data.csv'
df = pd.read_csv('insurance_data.csv')

# Display basic information about the dataset
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nDescriptive Statistics:\n", df.describe())

# Cell 3: Exploratory Data Analysis (EDA)
# Visualize the distribution of total charges
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
df['total_charges'].hist(bins=30)
plt.title('Distribution of Total Charges')
plt.xlabel('Total Charges')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.boxplot(x='smoker', y='total_charges', data=df)
plt.title('Total Charges by Smoking Status')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Cell 4: Data Preprocessing
# Handle missing values
df['bmi'].fillna(df['bmi'].median(), inplace=True)
df['smoker'].fillna(df['smoker'].mode()[0], inplace=True)

# Prepare features and target
X = df.drop('total_charges', axis=1)
y = df['total_charges']

# Preprocessing for numerical and categorical features
numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse=False), categorical_features)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Cell 5: Perceptron Model
def create_perceptron_model(input_dim):
    model = Sequential([
        Dense(1, input_dim=input_dim, activation='linear')
    ])
    return model

# Perceptron Model
perceptron_model = create_perceptron_model(X_train_processed.shape[1])
perceptron_model.compile(optimizer='sgd', loss='mean_squared_error')

# Train Perceptron
history_perceptron = perceptron_model.fit(
    X_train_processed, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_split=0.2, 
    verbose=0
)

# Cell 6: Deep Neural Network Models
def create_dnn_model(input_dim, layers=[32, 16, 8], learning_rate=0.01):
    model = Sequential()
    model.add(Dense(layers[0], input_dim=input_dim, activation='relu'))
    
    for neurons in layers[1:]:
        model.add(Dense(neurons, activation='relu'))
    
    model.add(Dense(1))  # Output layer
    
    optimizer = SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    return model

# Different DNN Configurations
dnn_configs = [
    {'layers': [32, 16, 8], 'lr': 0.01, 'batch_size': 32, 'epochs': 100},
    {'layers': [64, 32, 16], 'lr': 0.001, 'batch_size': 64, 'epochs': 150},
    {'layers': [128, 64, 32], 'lr': 0.0001, 'batch_size': 16, 'epochs': 200}
]

# Store model results
model_results = {}

for config in dnn_configs:
    # Create and train model
    model = create_dnn_model(
        input_dim=X_train_processed.shape[1], 
        layers=config['layers'], 
        learning_rate=config['lr']
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train_processed, y_train,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate model
    y_pred = model.predict(X_test_processed).flatten()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    model_results[str(config['layers'])] = {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'config': config
    }

# Cell 7: Optimizer Comparison
optimizers = [
    ('SGD', SGD(learning_rate=0.01)),
    ('Momentum', SGD(learning_rate=0.01, momentum=0.9)),
    ('Nesterov', SGD(learning_rate=0.01, momentum=0.9, nesterov=True)),
    ('Adam', Adam(learning_rate=0.001))
]

optimizer_results = {}

for name, optimizer in optimizers:
    model = create_dnn_model(input_dim=X_train_processed.shape[1])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    history = model.fit(
        X_train_processed, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    y_pred = model.predict(X_test_processed).flatten()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    optimizer_results[name] = {
        'mse': mse,
        'mae': mae,
        'r2': r2
    }

# Cell 8: Comparative Analysis and Visualization
# Visualize Model Performance
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(model_results.keys(), [res['mse'] for res in model_results.values()])
plt.title('MSE by Model Configuration')
plt.xlabel('Model Layers')
plt.ylabel('Mean Squared Error')

plt.subplot(1, 2, 2)
plt.bar(optimizer_results.keys(), [res['mse'] for res in optimizer_results.values()])
plt.title('MSE by Optimizer')
plt.xlabel('Optimizer')
plt.ylabel('Mean Squared Error')
plt.tight_layout()
plt.show()

# Print Results
print("DNN Model Configurations Results:")
for layers, result in model_results.items():
    print(f"\nLayers {layers}:")
    print(f"MSE: {result['mse']:.2f}")
    print(f"MAE: {result['mae']:.2f}")
    print(f"R2 Score: {result['r2']:.4f}")

print("\nOptimizer Comparison Results:")
for name, result in optimizer_results.items():
    print(f"\n{name} Optimizer:")
    print(f"MSE: {result['mse']:.2f}")
    print(f"MAE: {result['mae']:.2f}")
    print(f"R2 Score: {result['r2']:.4f}")

# Cell 9: Save Best Model
# Find the best model based on lowest MSE
best_model_layers = min(model_results, key=lambda x: model_results[x]['mse'])
best_model_config = model_results[best_model_layers]['config']

# Recreate and save the best model
best_model = create_dnn_model(
    input_dim=X_train_processed.shape[1], 
    layers=best_model_config['layers'], 
    learning_rate=best_model_config['lr']
)

best_model.fit(
    X_train_processed, y_train,
    epochs=best_model_config['epochs'],
    batch_size=best_model_config['batch_size'],
    verbose=0
)

# Save the model and preprocessor
best_model.save('insurance_cost_prediction_model.h5')
import joblib
joblib.dump(preprocessor, 'insurance_preprocessor.joblib')

print("Best model and preprocessor saved successfully!")
