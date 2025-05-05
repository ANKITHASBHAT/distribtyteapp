import pandas as pd

# Load preprocessed data
data = pd.read_csv("")

# Feature Engineering
# 1. Sales Velocity: Measures how fast a product sells
data["sales_velocity"] = data["quantity_sold"] / (data["lead_time_days"] + 1e-5)

# 2. Stockout Risk: Higher value indicates higher risk of running out of stock
data["stockout_risk"] = data["reorder_point"] / (data["demand_forecast"] + 1e-5)

# 3. Supplier Dependence: Helps understand reliance on suppliers
data["supplier_dependence"] = data["reorder_quantity"] / (data["quantity_sold"] + 1e-5)

# 4. High Demand Indicator: Binary feature indicating high demand
demand_mean = data["demand_forecast"].mean()
data["high_demand"] = (data["demand_forecast"] > demand_mean).astype(int)

# Save the engineered dataset
data.to_csv("engineered_data.csv", index=False)

print("Feature engineering complete. Engineered dataset saved as 'engineered_data.csv'.")
