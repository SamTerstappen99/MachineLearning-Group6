#price range prediction
#ML group 6 2024-04-18

#%% import libraries
import numpy as np 
import pandas as pd 
import seaborn as sns; sns.set(rc={'figure.figsize':(16,9)})
import matplotlib.pyplot as plt
from scipy import stats 
import folium
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
#import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


#%% import data and merge
olist_customers = pd.read_csv('olist_customers_dataset.csv')
olist_geolocation = pd.read_csv('olist_geolocation_dataset.csv')
olist_geolocation.drop_duplicates(subset="geolocation_zip_code_prefix", keep =  "last", inplace= True) #reduce duplicate coordinate under each post code
olist_orders = pd.read_csv('olist_orders_dataset.csv')
olist_items = pd.read_csv('olist_order_items_dataset.csv')
olist_order_payments = pd.read_csv('olist_order_payments_dataset.csv')
olist_reviews = pd.read_csv('olist_order_reviews_dataset.csv')
olist_products = pd.read_csv('olist_products_dataset.csv')
olist_sellers = pd.read_csv('olist_sellers_dataset.csv')
olist_category = pd.read_csv('product_category_name_translation.csv')

# Merge the datasets
olist_new = olist_orders.merge(olist_items, on='order_id', how='left')
olist_new = olist_new.merge(olist_order_payments, on='order_id', how='outer', validate='m:m')
olist_new = olist_new.merge(olist_reviews, on='order_id', how='outer')
olist_new = olist_new.merge(olist_products, on='product_id', how='outer')
olist_new = olist_new.merge(olist_customers, on='customer_id', how='outer')
olist_new = olist_new.merge(olist_sellers, on='seller_id', how='outer')
olist_new = olist_new.merge(olist_category, on='product_category_name', how='inner')
olist_new = olist_new.merge(olist_geolocation, "left", left_on=["customer_zip_code_prefix"], right_on=["geolocation_zip_code_prefix"]) # merge customer location into order

# Drop missing datetime related values
olist_new.dropna(subset= ['order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date'], inplace=True)
# Datetime features from Object to Datetime
olist_new['order_purchase_timestamp'] = pd.to_datetime(olist_new['order_purchase_timestamp'])
olist_new['order_delivered_customer_date'] = pd.to_datetime(olist_new['order_delivered_customer_date'])
olist_new['order_estimated_delivery_date'] = pd.to_datetime(olist_new['order_estimated_delivery_date'])
olist_new['shipping_limit_date'] = pd.to_datetime(olist_new['shipping_limit_date'])
olist_new['order_delivered_carrier_date'] =pd.to_datetime(olist_new['order_delivered_carrier_date'])
# Time-Stamps transition --> Converting Datetime from Object to Datetime
olist_new['purchase_year'] = olist_new['order_purchase_timestamp'].dt.year
olist_new['purchase_month'] = olist_new['order_purchase_timestamp'].dt.month
olist_new['purchase_day'] = olist_new['order_purchase_timestamp'].dt.day
olist_new['purchase_day_of_week'] = olist_new['order_purchase_timestamp'].dt.dayofweek
olist_new['purchase_hour'] = olist_new['order_purchase_timestamp'].dt.hour

olist_new.shape
olist_new.head()

#%% n_of_orders

# Number of orders per year
orders_per_year = olist_new['purchase_year'].value_counts().sort_index()
# Number of orders per month
orders_per_month = olist_new.groupby('purchase_month').size()
# Number of orders per day of the week
orders_per_day_of_week = olist_new['purchase_day_of_week'].value_counts().sort_index()
# Number of orders per hour
orders_per_hour = olist_new['purchase_hour'].value_counts().sort_index()

orders_per_month_category = olist_new.groupby(['purchase_month', 'product_category_name_english']).size().reset_index(name='order_count')
plt.figure(figsize=[12,8])
sns.lineplot(x=orders_per_month_category['purchase_month'].map({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}),
             y=orders_per_month_category['order_count'], hue=orders_per_month_category['product_category_name_english'])
plt.title('Number of Orders per Month by Product Category')
plt.xlabel('Month')
plt.ylabel('Number of Orders')
plt.legend(title='product_category_name_english', loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# %% data processing for regression
orders =  pd.read_csv("olist_orders_dataset.csv")
data_shane =  pd.read_csv("olist_data.csv")
data = pd.merge(orders, data_shane, on='order_id')
data.fillna(0, inplace=True)


data['product_category_name_english'] = data['product_category_name_english'].astype(str)
data['product_category_name_english_encoded'] = LabelEncoder().fit_transform(data['product_category_name_english'])
orders_per_month_category['product_category_name_english'] = orders_per_month_category['product_category_name_english'].astype(str)
label_encoder = LabelEncoder()
label_encoder.fit(data['product_category_name_english'])

orders_per_month_category['product_category_name_english_encoded'] = label_encoder.transform(orders_per_month_category['product_category_name_english'])
orders_per_month_category

#time of year
data['order_purchase_timestamp'] = pd.to_datetime(data['order_purchase_timestamp'])
data['purchase_month'] = data['order_purchase_timestamp'].dt.month
data['purchase_month'] = data['purchase_month'].apply(lambda x: x if x <= 12 else x - 12)

orders_per_month = pd.merge(orders_per_month_category, data, on=['product_category_name_english', 'purchase_month'], how='left')
data = pd.merge(data, orders_per_month_category, on=['product_category_name_english', 'purchase_month'], how='left')


#%% location and price range cluster
geolocaties = data[['geolocation_lat', 'geolocation_lng']].values
aantal_clusters = 100
kmeans = KMeans(n_clusters=aantal_clusters, random_state=0)
kmeans.fit(geolocaties)
data['location_cluster'] = kmeans.labels_

import folium
from sklearn.cluster import KMeans

# make a map with folim
map_brazil = folium.Map(location=[-14.235, -51.925], zoom_start=4)


for cluster_label in data['location_cluster'].unique():
    cluster_data = data[data['location_cluster'] == cluster_label]
    cluster_center = cluster_data[['geolocation_lat', 'geolocation_lng']].mean()
    folium.Marker(location=[cluster_center[0], cluster_center[1]], popup=f"Cluster {cluster_label}").add_to(map_brazil)

#show the map
map_brazil

#Cluster of prices
price_cluster = data[['price']].values
n = 75
kmeans = KMeans(n_clusters=n, random_state=0)
kmeans.fit(price_cluster)
data['price_cluster'] = kmeans.labels_

print('Total amount of clusters', data['price_cluster'].max() + 1)

prijsranges = [(cluster, data[data['price_cluster'] == cluster]['price'].min(), data[data['price_cluster'] == cluster]['price'].max()) for cluster in range(data['price_cluster'].max() + 1)]
prijsranges_sorted = sorted(prijsranges, key=lambda x: x[1])

# print("Lijst van alle clusters en de prijsranges (gesorteerd op laagste prijs):")
for cluster, min_prijs, max_prijs in prijsranges_sorted:
    print(f"Cluster {cluster}: Pricerange {min_prijs} - {max_prijs}")


#%% plot price range
# Sort clusters pricerange

sorted_ranges = sorted(prijsranges_sorted, key=lambda x: x[1])
clusters, min_prices, max_prices = zip(*sorted_ranges)

# Bereken het aantal items in elk prijscluster
cluster_counts = [data[data['price_cluster'] == cluster].shape[0] for cluster in clusters]

# Plot het aantal items per prijscluster
plt.figure(figsize=(20, 9))
plt.bar(range(len(clusters)), cluster_counts, color='r', alpha=0.8)
plt.xlabel('Price Cluster')
plt.ylabel('Number of items')
plt.title('Number of items per cluster')
plt.xticks(range(len(clusters)), clusters)
plt.grid(True)
plt.show()


#%% Regression
mean_order_count = data['order_count'].mean()
# remove na
data['order_count'].fillna(mean_order_count, inplace=True)
product_categories = len(data['product_category_name_english'].unique())
print("Number of unique product categories:", product_categories)

X = data[['product_category_name_english_encoded_x','purchase_month','location_cluster','order_count' ]]  # Selecteer kenmerken

y = data['price_cluster']  # Selecteer label

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model with increased max_iter
model = LogisticRegression(max_iter=1000)  # You can adjust the number of iterations as needed

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

#test sample
sample_indices = np.random.choice(len(X_test), size=10, replace=False)  # Randomly select 10 examples from the test data
X_sample = X_test.iloc[sample_indices]
y_sample_true = y_test.iloc[sample_indices]
y_sample_pred = model.predict(X_sample)

# Show the actual and predicted clusters
for i in range(len(sample_indices)):
    print(f"Example {i}:")
    print(f"Actual cluster: {y_sample_true.iloc[i]}, Predicted cluster: {y_sample_pred[i]}")


#%% seller interaction
#Ask the user to enter the month, price range and location cluster
purchase_month = int(input("What month do you want to sell (indicate with a number): "))
product_category_name_english = int(input("What is the product category name?: "))
location_cluster = int(input("In which location cluster you intend to sell?: "))

# Filter de DataFrame orders_per_month_category op basis van de ingevoerde waarden
filtered_orders = orders_per_month_category[
    (orders_per_month_category['purchase_month'] == purchase_month) & 
    (orders_per_month_category['product_category_name_english_encoded'] == product_category_name_english)
]

#Convert the entered values ​​into a numpy array with the correct shape (1 by 3 array)
X_array = np.array([[purchase_month, product_category_name_english, location_cluster, filtered_orders.iloc[0]['order_count']]])

predicted_price_cluster = model.predict(X_array)

print("Predicted price cluster:", predicted_price_cluster)

# Find the price range that matches the predicted price cluster
predicted_cluster = int(predicted_price_cluster[0])  # Because 'predicted_price_cluster' is probably an array
for cluster, min_prijs, max_prijs in prijsranges_sorted:
    if cluster == predicted_cluster:
        print(f"Price range of predicted cluster {predicted_cluster}: {min_prijs} - {max_prijs}")
        break

#%% Ask the user to enter the necessary information
purchase_month = int(input("What month do you want to sell (indicate with a number): "))
product_category_code = int(input("What is the product category code?: "))

# Filter the DataFrame orders_per_month_category based on the entered values
filtered_orders = orders_per_month_category[
    (orders_per_month_category['purchase_month'] == purchase_month) & 
    (orders_per_month_category['product_category_name_english_encoded'] == product_category_code)
]

# Check if a matching row was found
if not filtered_orders.empty:
    # Get the order_count from the first corresponding row
    order_count = filtered_orders.iloc[0]['order_count']
    print(f"The order count for the specified month and product category is: {order_count}")
else:
    print("No data was found for the specified month and product category.")

# %%
