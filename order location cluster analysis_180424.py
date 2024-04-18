#order location cluster analysis
#ML group 6 at 2024-04-18

#%% import packages and data

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data =  pd.read_csv("olist_data.csv") # premerged dataset

#%% group location based on city
mean_location = data.groupby('geolocation_city')[['geolocation_lat', 'geolocation_lng']].mean()
total_per_city = data.groupby('geolocation_city')['price'].sum()
mean_location['total_price'] = total_per_city
mean_location = mean_location.reset_index()

# Show the result
mean_location


#%% location clusters
geolocaties = mean_location[['geolocation_lat', 'geolocation_lng']].values

n_clusters = 4200

#K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(geolocaties)

# assign clusterlabels to DataFrame
mean_location['cluster_label'] = kmeans.labels_
# Set  clusterlabels as index

# Show the result
mean_location


#%% group data based on cluster labels
grouped = mean_location.groupby('cluster_label')

# different operators for cols
cluster_stats = grouped.agg({
    'geolocation_lat': 'mean',  # mean geolocation_lat
    'geolocation_lng': 'mean',  # mean geolocation_lng
    'total_price': 'sum'        #sum of price
})

# Show the result
print(cluster_stats)



#%% Plot 3d figures
fig = plt.figure(figsize=(120, 8))
ax = fig.add_subplot(111, projection='3d')

# Loop over each cluster and draw a bar of total_price on plane xy
for index, row in cluster_stats.iterrows():
    ax.bar3d(row['geolocation_lng'], row['geolocation_lat'], 0, 1, 1, row['total_price'], color='skyblue')

#axis label
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Total Price')

# plot title
plt.title('Total Price per Cluster in XY Plane')

#show the plot
plt.show()


#%% 2d plot
# define area of brazil
brazil_border = {
    'lon': [-75, -30],
    'lat': [-35, 5]
}

plt.figure(figsize=(15, 10)) 

# plot brazil border
plt.plot(brazil_border['lon'], [brazil_border['lat'][0]]*2, color='black')
plt.plot(brazil_border['lon'], [brazil_border['lat'][1]]*2, color='black')
plt.plot([brazil_border['lon'][0]]*2, brazil_border['lat'], color='black')
plt.plot([brazil_border['lon'][1]]*2, brazil_border['lat'], color='black')

# Plot de clusters on the map
for index, row in cluster_stats.iterrows():
    cluster_position = (row['geolocation_lng'], row['geolocation_lat'])
    marker_size = row['total_price'] / 50000  
    plt.scatter(*cluster_position, color='blue', s=marker_size, alpha=0.7)

#axis label
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# plot title
plt.title('Total Price per Cluster on Map of Brazil')

#set plot ratio
plt.gca().set_aspect('equal', adjustable='box')

#show the plot
plt.show()

