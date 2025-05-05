import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense

# 🔹 Load feature-engineered data
data = pd.read_csv("/Users/amruthabhat/Documents/demand_forecasting/src/engineered_data.csv")

# 🔹 Define features and target
TARGET_COLUMN = "demand_forecast"
features = data.columns.tolist()
features.remove("product_id")  # Removing non-numeric identifier
features.remove("product_name")  # Removing categorical text column
features.remove("category")  # Removing categorical text column
features.remove("supplier_id")  # Removing categorical text column
features.remove(TARGET_COLUMN)

X = data[features]
y = data[TARGET_COLUMN]

# 🔹 Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 🔹 Reshape Data for BiLSTM (3D: [samples, timesteps, features])
def reshape_for_lstm(X):
    return np.expand_dims(X.values, axis=1)  # Adding time-step dimension

X_train_lstm = reshape_for_lstm(X_train)
X_val_lstm = reshape_for_lstm(X_val)
X_test_lstm = reshape_for_lstm(X_test)

# 🔹 Define BiLSTM Model
bi_lstm_model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, activation='relu')),
    Bidirectional(LSTM(32, activation='relu')),
    Dense(1)
])

bi_lstm_model.compile(optimizer='adam', loss='mse')

# 🔹 Train BiLSTM
bi_lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=32, validation_data=(X_val_lstm, y_val))

# 🔹 Extract Features from BiLSTM
lstm_features_train = bi_lstm_model.predict(X_train_lstm).flatten()
lstm_features_test = bi_lstm_model.predict(X_test_lstm).flatten()

# 🔹 NARX Implementation using PyTorch
class NARXModel(nn.Module):
    def __init__(self, input_size):
        super(NARXModel, self).__init__()
        self.hidden_layer = nn.Linear(input_size + 1, 32)  # +1 for previous output (auto-regressive)
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x, prev_output):
        x = torch.cat((x, prev_output), dim=1)  # Concatenating exogenous features
        x = torch.relu(self.hidden_layer(x))
        return self.output_layer(x)

# 🔹 Train NARX Model
input_size = X_train.shape[1]
narx_model = NARXModel(input_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(narx_model.parameters(), lr=0.01)

X_train_torch = torch.tensor(X_train.values, dtype=torch.float32)
y_train_torch = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

prev_output = torch.zeros_like(y_train_torch)  # Initial previous output
for epoch in range(100):
    optimizer.zero_grad()
    y_pred_narx = narx_model(X_train_torch, prev_output)
    loss = criterion(y_pred_narx, y_train_torch)
    loss.backward()
    optimizer.step()
    prev_output = y_pred_narx.detach()

# 🔹 Generate NARX Features for Training
X_train_torch = torch.tensor(X_train.values, dtype=torch.float32)
prev_output_train = torch.zeros((X_train.shape[0], 1), dtype=torch.float32)
y_pred_narx_train = narx_model(X_train_torch, prev_output_train).detach().numpy().flatten()

# 🔹 Generate NARX Features for Testing
X_test_torch = torch.tensor(X_test.values, dtype=torch.float32)
prev_output_test = torch.zeros((X_test.shape[0], 1), dtype=torch.float32)
y_pred_narx_test = narx_model(X_test_torch, prev_output_test).detach().numpy().flatten()


X_train_combined = np.column_stack((X_train, lstm_features_train, y_pred_narx_train))
X_test_combined = np.column_stack((X_test, lstm_features_test, y_pred_narx_test))

# 🔹 Train XGBoost Model
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_reg.fit(X_train_combined, y_train)

# 🔹 Predict and Evaluate
y_pred = xgb_reg.predict(X_test_combined)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse:.4f}, R² Score: {r2:.4f}')

# 🔹 Save Models
bi_lstm_model.save('bi_lstm_model.h5')  # Save BiLSTM properly

torch.save(narx_model.state_dict(), 'narx_model.pth')  # Save NARX correctly
joblib.dump(xgb_reg, 'xgboost_model.pkl')  # Save XGBoost model


print('Hybrid Model (BiLSTM + NARX + XGBoost) saved successfully!')
