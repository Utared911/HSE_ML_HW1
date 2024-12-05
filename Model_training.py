from sklearn.linear_model import LinearRegression
import joblib
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from Data_preprocessing_for_training import extract_mileage, extract_engine, extract_max_power, extract_torque_RPM, seats2int, name2brand, remove_duplicates, train_encoder_names, encode_names
import pandas as pd
import numpy as np

df_train = pd.read_csv('/Users/mac/opt/PycharmProject/HW1_HSE_ML/pythonProject/cars_train_local.csv')
df_test = pd.read_csv('/Users/mac/opt/PycharmProject/HW1_HSE_ML/pythonProject/cars_test_local.csv')
df_train = df_train.drop(columns=['fuel', 'seller_type', 'transmission', 'owner'])
df_test = df_test.drop(columns=['fuel', 'seller_type', 'transmission', 'owner'])
# print(df_train.head(10))
def preprocess_dataframe(df):
    df = df.copy()

    # Remove duplicates
    df = remove_duplicates(df)

    # Handle 'mileage', 'engine', and 'max_power'
    df['mileage'] = df['mileage'].apply(extract_mileage)
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')

    df['engine'] = df['engine'].apply(extract_engine)
    df['engine'] = pd.to_numeric(df['engine'], errors='coerce')

    df['max_power'] = df['max_power'].apply(extract_max_power)
    df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')

    torque_rpm = df['torque'].apply(lambda x: pd.Series(extract_torque_RPM(x)))
    df['torque'] = torque_rpm[0]
    df['rpm'] = torque_rpm[1]
    df['torque'] = pd.to_numeric(df['torque'], errors='coerce')
    df['rpm'] = pd.to_numeric(df['rpm'], errors='coerce')

    df['seats'] = df['seats'].apply(seats2int)
    df['seats'] = pd.to_numeric(df['seats'], errors='coerce')

    df['name'] = df['name'].apply(name2brand)
    return df

def business_metrics(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    deviation = np.abs(y_pred - y_true) / y_true
    within_10_percent = np.sum(deviation <= 0.1)
    total_predictions = len(y_true)
    return within_10_percent / total_predictions

df_train = preprocess_dataframe(df_train)
df_test = preprocess_dataframe(df_test)
encoder = train_encoder_names(df_train)
df_train = encode_names(df_train, encoder)
df_test = encode_names(df_test, encoder)
# print(df_train.info())
# print(df_train.describe().T)
# print(df_train.head(10))

y_train = df_train['selling_price']
X_train = df_train.drop(columns=['selling_price'])

y_test = df_test['selling_price']
X_test = df_test.drop(columns=['selling_price'])

brand_columns = [col for col in X_train.columns if col.startswith("brand_")]
numeric_columns = [col for col in X_train.columns if col not in brand_columns]

scaler = StandardScaler()

X_train_scaled_numeric = scaler.fit_transform(X_train[numeric_columns])
# print(X_train[numeric_columns].head())
X_test_scaled_numeric = scaler.transform(X_test[numeric_columns])

X_train_scaled = pd.concat([
    pd.DataFrame(X_train_scaled_numeric, columns=numeric_columns, index=X_train.index),
    X_train[brand_columns]
], axis=1)

X_test_scaled = pd.concat([
    pd.DataFrame(X_test_scaled_numeric, columns=numeric_columns, index=X_test.index),
    X_test[brand_columns]
], axis=1)

model = LinearRegression()

model.fit(X_train_scaled, y_train)

y_test_pred = model.predict(X_test_scaled)

r2_test = r2_score(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
business_metric = business_metrics(y_test, y_test_pred)

print(f"Test R^2: {r2_test:.4f}, Test MSE: {mse_test:.4f}, Test RMSE: {rmse_test:.4f}")
print(f"Test Business Metric (Within 10%): {business_metric:.4f}")

# Save the trained model
model_filename = "linear_regression_model_cars_Alan.pkl"
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

# Save the scaler
scaler_filename = "scaler.pkl"
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved to {scaler_filename}")

# Save the encoder
encoder_filename = "encoder.pkl"
joblib.dump(encoder, encoder_filename)
print(f"Encoder saved to {encoder_filename}")