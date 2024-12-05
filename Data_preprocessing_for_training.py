from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
import re
import math

# Constants
kgm_to_nm = 9.80665
kg_to_liters = 1.39
Median_year = 2014.0
Median_km_driven = 70000.0
Median_mileage_kmpl = 19.4
Median_engine_СС = 1248.0
Median_max_power_bph = 81.86
Median_seats = 5.0
Median_torque_nm = 160.0
Median_rpm = 3000.0

columns_for_train = ['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type',
       'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque',
       'seats']

columns_for_inference = ['name', 'year', 'km_driven', 'fuel', 'seller_type',
       'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque',
       'seats']

brands = [
    "Maruti",
    "Hyundai",
    "Mahindra",
    "Tata",
    "Toyota",
    "Ford",
    "Honda",
    "Chevrolet",
    "Renault",
    "Volkswagen",
    "Nissan",
    "Skoda",
    "Datsun",
    "Mercedes-Benz",
    "BMW",
    "Fiat",
    "Audi"]

# Function for processing mileage
def extract_mileage(value):
    if value == None:
        return Median_mileage_kmpl
    if value is float or value is int:
        return value
    elif pd.isna(value):
        return Median_mileage_kmpl
    try:
        units = value.split()[1]
        numeric_part = float(value.split()[0])
        if units == "km/kg":
            numeric_part = float(value.split()[0]) * kg_to_liters
        return numeric_part
    except (AttributeError, ValueError, IndexError):
        return Median_mileage_kmpl

# Function for processing 'max_power'
def extract_max_power(value):
    if value == None:
        return Median_max_power_bph
    if value is float or value is int:
        if value == 0:
            return Median_max_power_bph
        else:
            return value
    try:
        return float(value.split()[0])
    except (AttributeError, ValueError, IndexError):
        return Median_max_power_bph

# Function for processing 'engine'
def extract_engine(value):
    if value == None:
        return Median_engine_СС
    if value is float or value is int:
        if value == 0:
            return Median_engine_СС
        else:
            return value
    try:
        return float(value.split()[0])
    except (AttributeError, ValueError, IndexError):
        return Median_engine_СС

# Function for processing 'torque'
def extract_torque_RPM(value):
    if value is None or not isinstance(value, str):
        return Median_torque_nm, Median_rpm

    # Normalize the input string
    value = re.sub(r'\s+', ' ', value.strip()).lower()
    value = value.replace(",", "")

    # Initialize default values
    torque_value = Median_torque_nm
    rpm_value = Median_rpm
    torque_unit = 'nm'

    # Match cases like "190Nm" or "22.4kgm"
    torque_match1 = re.search(r'(\d+\.?\d*)\s?(nm|kgm)', value)

    # Match cases like "11.5@ 4500(kgm@ rpm)"
    torque_match2 = re.search(r'(\d+\.?\d*)\s?@\s?(\d+(?:-\d+)?)(?:,\d+)?\((kgm|nm)\s?@\s?rpm\)', value)

    if torque_match1:
        torque_value = float(torque_match1.group(1))
        torque_unit = torque_match1.group(2)
        if torque_unit == 'kgm':
            torque_value = torque_value * kgm_to_nm

        rpm_match = re.search(r'(\d+(?:-\d+)?)\s?rpm', value)
        if rpm_match:
            rpm_range = rpm_match.group(1)
            rpm_value = float(rpm_range.split('-')[-1])

    elif torque_match2:
        torque_value = float(torque_match2.group(1))
        rpm_range = torque_match2.group(2)
        rpm_value = float(rpm_range.split('-')[-1])
        torque_unit = torque_match2.group(3)

        if torque_unit == 'kgm':
            torque_value = torque_value * kgm_to_nm

    if torque_value == 0:
        torque_value = Median_torque_nm
    if rpm_value == 0:
        rpm_value = Median_rpm

    return torque_value, rpm_value

# Function to treansform seats
def seats2int(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return Median_seats
    try:
        return int(value)
    except ValueError:
        return Median_seats

# Function to handle duplicated cars
def remove_duplicates(df):
    features_only = df.drop(columns=['selling_price'])
    df = df.drop_duplicates(subset=features_only.columns, keep='first')
    df.reset_index(drop=True, inplace=True)
    return df

# Function to handle omissions

# Function to transform name to brand
def name2brand(value):
    for brand in brands:
        if brand in value:
            return brand
        else:
            continue
    return 'Rare'

# Function to train encoder Brands
def train_encoder_names(df):
    categories = [df['name'].unique().tolist()]
    encoder = OneHotEncoder(categories=categories, drop=None, sparse_output=False)
    encoder.fit(df[['name']])
    return encoder

# fonction to encode names
def encode_names(df, encoder):
    df_encoded = encoder.transform(df[['name']])
    encoded_df = pd.DataFrame(
        df_encoded,
        columns=[f"brand_{cat}" for cat in encoder.categories_[0]],
        index=df.index)
    encoded_df = encoded_df.drop(columns=['brand_Rare'])
    return pd.concat([df.drop(columns=['name']), encoded_df], axis=1)

