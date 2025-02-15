import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random

ratings_file = "attribute_ratings.csv"
metrics_file = "v4_face_perception_metrics.csv"

ratings_df = pd.read_csv(ratings_file)
metrics_df = pd.read_csv(metrics_file)

ratings_df['stimulus'] = ratings_df['stimulus'].astype(str)
metrics_df['Image Name'] = metrics_df['Image Name'].str.replace('.jpg', '', regex=False)

metrics_columns = [col for col in metrics_df.columns if col != 'Image Name']
metrics_df_reduced = metrics_df[['Image Name'] + metrics_columns]

ratings_df['stimulus'] = ratings_df['stimulus'].astype(str)
metrics_df['Image Name'] = metrics_df['Image Name'].str.replace('.jpg', '', regex=False)

metrics_columns = [col for col in metrics_df.columns if col != 'Image Name']
metrics_df_reduced = metrics_df[['Image Name'] + metrics_columns]


# merging rating and metrics
merged_df = pd.merge(
    ratings_df[['rating', 'stimulus', 'attribute']], 
    metrics_df_reduced, 
    left_on='stimulus', 
    right_on='Image Name', 
    how='inner'
)
merged_df = merged_df.dropna()



# finding the mean for every unique attribute

mean_df = merged_df.pivot_table(index='Image Name', columns='attribute', values='rating', aggfunc='mean')
mean_df.columns = [f"mean_{col}" for col in mean_df.columns]  
mean_df.reset_index(inplace=True) 

# finding the mean of the rating column 
mean_ratings = merged_df.groupby('Image Name')['rating'].mean().reset_index()
mean_ratings.rename(columns={'rating': 'mean_rating'}, inplace=True)
mean_df = mean_df.merge(mean_ratings, on='Image Name', how='left')


merged_df = merged_df.merge(mean_df, on='Image Name', how='left')
merged_df.head()


# 80/20 split of unique images
unique_images = merged_df['Image Name'].unique()
random.seed(42)
train_images = random.sample(list(unique_images), k=int(0.8 * len(unique_images)))
test_images = [img for img in unique_images if img not in train_images]

train_df = merged_df[merged_df['Image Name'].isin(train_images)]
test_df = merged_df[merged_df['Image Name'].isin(test_images)]


train_df = merged_df[merged_df['Image Name'].isin(train_images)]
test_df = merged_df[merged_df['Image Name'].isin(test_images)]

all_features = [
    'Philtrum Width', 
    'Labio-oral region', 
    'Nose Width', 
    'lateral upper lip heights (left and right)', 
    'eye fissure height (left and right)','Average_ITA', 'edge_of_similarity_average', 'Upper Vermilion Height', 'Lower Face Height',
    'Lower Vermilion Height', 'Upper Lip Height', 'Columella Length', 'orbit and brow height (left and right)', 'eye fissure height (left and right)', 
    'Intercanthal Face Height', 'Labio-oral region', 'Nose Height', 'Orbits Biocular Width', 'Orbits Fissure Length(left and right)', 'Orbits Intercanthal Width',
    'Face Width 2', 'Face Width', 'Face Height 3', 'Face Height 2', 'Face Height', 'Head Height',
]

# modeling for all unique attributes

enet_results = []
mean_df = mean_df.drop(columns = ["Image Name"])

for idx, mean in enumerate(mean_df.columns):
    
    
    if mean not in train_df.columns:
        print(f"Skip {mean}  not in train_df!")
        continue

    # predicting the mean of all unique attributes
    X_train = train_df[all_features]
    y_train = train_df[mean]
    X_test = test_df[all_features]
    y_test = test_df[mean]

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train ElasticNet model
    enet = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
    enet.fit(X_train_scaled, y_train)

    y_pred = enet.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    
    
    enet_results.append({
            "Target Variable": mean,
            "RMSE": rmse,
            "R² Score": r2,
        })


    results_df = pd.DataFrame(enet_results)
    results_df.to_csv("elastic net results.csv", index=False)
    