# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import random

# Load the data
ratings_file = "attribute_ratings.csv"
metrics_file = "v4_face_perception_metrics.csv"

ratings_df = pd.read_csv(ratings_file)
metrics_df = pd.read_csv(metrics_file)
face_features = metrics_df.columns[1:]

# Preprocess metrics
metrics_df['Image Name'] = metrics_df['Image Name'].str.replace('.jpg', '', regex=False)
ratings_df['stimulus'] = ratings_df['stimulus'].astype(str)

# Merge ratings with metrics
merged_df = pd.merge(
    ratings_df[['rating', 'stimulus', 'attribute']],
    metrics_df,
    left_on='stimulus',
    right_on='Image Name',
    how='inner'
)
merged_df = merged_df.dropna()

# Train/test split on individual ratings
unique_images = merged_df['Image Name'].unique()
random.seed(42)
train_images = random.sample(list(unique_images), k=int(0.8 * len(unique_images)))
test_images = [img for img in unique_images if img not in train_images]

train_df = merged_df[merged_df['Image Name'].isin(train_images)]
test_df = merged_df[merged_df['Image Name'].isin(test_images)]

# Model training and prediction for each attribute
enet_results = []

for attribute in merged_df['attribute'].unique():
    train_subset = train_df[train_df['attribute'] == attribute]
    test_subset = test_df[test_df['attribute'] == attribute]

    if train_subset.empty or test_subset.empty:
        continue

    X_train = train_subset[face_features]
    y_train = train_subset['rating']
    X_test = test_subset[face_features]
    y_test = test_subset['rating']

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Ridge model
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)

    # Predict ratings for test set
    test_subset['predicted_rating'] = ridge.predict(X_test_scaled)

    # Calculate metrics for mean predictions
    mean_actual = test_subset.groupby('Image Name')['rating'].mean()
    mean_predicted = test_subset.groupby('Image Name')['predicted_rating'].mean()

    mse = mean_squared_error(mean_actual, mean_predicted)
    r2 = r2_score(mean_actual, mean_predicted)
    rmse = np.sqrt(mse)

    enet_results.append({
        "Attribute": attribute,
        "RMSE": rmse,
        "R² Score": r2
    })

# Save the results
results_df = pd.DataFrame(enet_results)
results_df.to_csv("individual_ratings_mean_predictions.csv", index=False)
print(results_df) 

