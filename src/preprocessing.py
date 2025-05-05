import pandas as pd
import os
import logging
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load data
data = pd.read_csv('/Users/amruthabhat/Documents/demand_forecasting/data/sample_data.csv')

# Handle missing values
data.fillna(method='ffill', inplace=True)  # Forward fill for categorical and general missing values
data['lead_time_days'].fillna(0, inplace=True)  # Replace missing lead time with 0

# Remove duplicates
data.drop_duplicates(inplace=True)

# Encoding categorical features
categorical_features = ['category', 'supplier_id']
ordinal_encoder = OrdinalEncoder()
data[categorical_features] = ordinal_encoder.fit_transform(data[categorical_features])

# Normalizing numerical features
scaler = MinMaxScaler()
numeric_features = ['unit_price', 'quantity_sold', 'demand_forecast', 'lead_time_days', 'reorder_point', 'reorder_quantity']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Splitting data
train_data, test_data = train_test_split(data, test_size=0.15, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42)

# Ensure processed data directory exists
processed_data_dir = "data/processed"
os.makedirs(processed_data_dir, exist_ok=True)

# Save processed data
train_data.to_csv(f"{processed_data_dir}/train_data.csv", index=False)
val_data.to_csv(f"{processed_data_dir}/val_data.csv", index=False)
test_data.to_csv(f"{processed_data_dir}/test_data.csv", index=False)

logging.info("Preprocessing complete. Train, validation, and test sets saved in 'data/processed/' directory.")
