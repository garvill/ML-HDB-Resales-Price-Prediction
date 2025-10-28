import pandas as pd
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


files = glob.glob('C:/Users/user/Downloads/wj_test/2. Real Estate Analysis - Resale of Singapore HDB Flats/Datasets/*.csv')  # replace with your folder path

dfs = []
for f in files:
    df = pd.read_csv(f)
    # Drop 'remaining_lease' if it exists
    if 'remaining_lease' in df.columns:
        df = df.drop(columns=['remaining_lease'])
    dfs.append(df)


data = pd.concat(dfs, ignore_index=True, sort=False)

#print(data.shape)

# Drop duplicates
data = data.drop_duplicates()
#print(data.shape)
#print(data.columns)
# Standardize column names
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

#Drop any column that have missing value
data = data.sort_values(by='lease_commence_date')

#Convert lease commence date to numbers only
data['lease_commence_date'] = pd.to_numeric(data['lease_commence_date'], errors='coerce')
data['remaining_lease'] = 99 - (2025 - data['lease_commence_date'])

# Convert to datetime
data['month'] = pd.to_datetime(data['month'], errors='coerce')

# Extract year and month as separate numeric columns
data['year'] = data['month'].dt.year
data['month_num'] = data['month'].dt.month

# Drop original month column
data = data.drop(columns=['month'])

data = data.dropna()
# List of categorical columns to encode
categorical_cols = ['town', 'flat_type', 'block', 'street_name', 'storey_range', 'flat_model']

# Apply LabelEncoder to each categorical column

label_encoders = {}  # store encoders for later use
for col in categorical_cols:
    # Convert to lowercase strings
    data[col] = data[col].astype(str).str.lower()
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# print(data.tail(10))


# plt.scatter(data['remaining_lease'], data['resale_price'], alpha=0.5)
# plt.title('Resale Price vs Remaining Lease')
# plt.xlabel('Remaining Lease')
# plt.ylabel('Resale Price')


# # Reverse x-axis
# plt.gca().invert_xaxis()  # gca = get current axis

# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()


# Separate features and target
X = data.drop('resale_price', axis=1)  # all columns except target
y = data['resale_price']               # target column

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,   # number of trees
    random_state=42,
    n_jobs=-1           # use all CPU cores
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# print("MAE:", mae)
# print("RMSE:", rmse)
# print("R²:", r2)



feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# print(feature_importances)


# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title("Predicted vs Actual Resale Prices")
# plt.show()


def predict_price(flat_info):
    """
    Predict resale price for HDB flats.
    
    flat_info: dict or pandas DataFrame
        Can include only some columns. Missing columns will be filled safely.
         
    Returns:
        Predicted price (float) or array of prices
    """
    import pandas as pd
    import numpy as np

    # Convert dict to DataFrame if needed
    if isinstance(flat_info, dict):
        flat_info = pd.DataFrame([flat_info])

    # Calculate remaining lease if lease_commence_date is provided
    if 'remaining_lease' not in flat_info.columns and 'lease_commence_date' in flat_info.columns:
        flat_info['remaining_lease'] = 99 - (2025 - flat_info['lease_commence_date'])

    # Fill missing numeric columns with median from training
    for col in X_train.select_dtypes(include=[np.number]).columns:
        if col not in flat_info.columns:
            flat_info[col] = X_train[col].median()

    # Handle categorical columns
    categorical_cols = ['town', 'flat_type', 'block', 'street_name', 'storey_range', 'flat_model']
    for col in categorical_cols:
        if col in flat_info.columns:
            flat_info[col] = flat_info[col].astype(str).str.lower()
            le = label_encoders[col]
            # Map unseen labels to most frequent known category
            flat_info[col] = flat_info[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else np.argmax(np.bincount(le.transform(le.classes_)))
            )
        else:
            # Missing categorical → most frequent value
            le = label_encoders[col]
            flat_info[col] = np.argmax(np.bincount(le.transform(le.classes_)))

    # Ensure all columns are present and in the correct order
    flat_info = flat_info[X_train.columns]

    # Predict
    predicted_price = model.predict(flat_info)
    return predicted_price
