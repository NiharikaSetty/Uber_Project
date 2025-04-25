from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import folium
from datetime import datetime
from scipy.optimize import linear_sum_assignment
import requests
import random
import matplotlib.pyplot as plt
import io
import base64
from math import radians, cos, sin, sqrt, atan2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

df = pd.read_csv("uber_rides.csv")
geolocator = Nominatim(user_agent="smart_uber_system")

def geocode_address(address, max_retries=3):
    address = address.strip()
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(address, timeout=10)
            if location:
                return location.latitude, location.longitude
        except GeocoderTimedOut:
            continue
    return None, None

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)*2 + cos(lat1) * cos(lat2) * sin(dlon / 2)*2
    a = max(0, min(1, a))
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    r = 6371
    return c * r

def estimate_fare(distance, weather):
    base_fare = 15
    per_km_rate = 1
    surge = 1.0
    if weather == "Rainy":
        surge = 1.2
    elif weather == "Sunny":
        surge = 1.0
    elif weather == "Stormy":
        surge = 1.5
    return round((base_fare + distance * per_km_rate) * surge, 2)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/driver-details')
def driver_details():
    driver_df = pd.read_csv('driver_details.csv')
    random_driver = driver_df.sample(n=1).to_dict(orient='records')[0]
    return render_template('driver_details.html', driver=random_driver)

@app.route('/rider', methods=['GET', 'POST'])
def rider():
    if request.method == 'POST':
        pickup_address = request.form['pickup_address']
        drop_address = request.form['drop_address']
        weather = request.form['weather']

        if weather not in ["Clear", "Rainy", "Stormy", "Sunny"]:
            return "Invalid weather input."

        pickup_lat, pickup_lon = geocode_address(pickup_address)
        drop_lat, drop_lon = geocode_address(drop_address)

        if pickup_lat is None or drop_lat is None:
            return "Could not geocode one or both addresses."

        osrm_url = f"http://router.project-osrm.org/route/v1/driving/{pickup_lon},{pickup_lat};{drop_lon},{drop_lat}"
        params = {
            "overview": "full",
            "alternatives": "true",
            "geometries": "geojson"
        }
        response = requests.get(osrm_url, params=params)
        routes = response.json().get("routes", [])

        if not routes:
            return "No route found between the given addresses."

        optimal_route = routes[0]
        distance_km = optimal_route["distance"] / 1000

        driver_df = df[['driver_id', 'pickup_lat', 'pickup_lon']].drop_duplicates()
        driver_df.dropna(subset=['pickup_lat', 'pickup_lon'], inplace=True)

        driver_df['distance'] = driver_df.apply(
            lambda row: haversine(pickup_lat, pickup_lon, row['pickup_lat'], row['pickup_lon']), axis=1
        )
        nearest_driver = driver_df.sort_values(by='distance').iloc[0]

        driver_details = {
            "driver_id": nearest_driver['driver_id'],
            "name": f"Driver {nearest_driver['driver_id']}",
            "rating": round(np.random.uniform(4.0, 5.0), 2),
            "phone": f"+91-9{random.randint(100000000, 999999999)}"
        }

        fare = estimate_fare(distance_km, weather)

        route_map = folium.Map(location=[pickup_lat, pickup_lon], zoom_start=13)
        folium.Marker([pickup_lat, pickup_lon], popup="Pickup", icon=folium.Icon(color='green')).add_to(route_map)
        folium.Marker([drop_lat, drop_lon], popup="Drop", icon=folium.Icon(color='red')).add_to(route_map)

        for i, route in enumerate(routes):
            coords = route["geometry"]["coordinates"]
            path = [[lat, lon] for lon, lat in coords]
            color = 'blue' if i == 0 else 'black'
            folium.PolyLine(locations=path, color=color, weight=5, opacity=0.8).add_to(route_map)

        route_map.save('static/ride_route.html')

        return render_template('result.html',
                               pickup_address=pickup_address,
                               drop_address=drop_address,
                               distance_km=round(distance_km, 2),
                               fare=fare,
                               weather=weather,
                               driver=driver_details,
                               show_map=True)

    return render_template('rider.html')

@app.route("/random_driver")
def random_driver():
    df = pd.read_csv("uber_rides.csv")
    random_row = df.sample(1).iloc[0]
    return jsonify({
        "driver_id": random_row["driver_id"],
        "name": f"Driver {random_row['driver_id']}",
        "rating": round(random.uniform(4.0, 5.0), 2),
        "phone": f"+91-9{random.randint(100000000, 999999999)}"
    })

@app.route('/admin')
def admin():
    df = pd.read_csv('uber_rides.csv')
    df['pickup_time'] = pd.to_datetime(df['pickup_time'], dayfirst=True)
    df['date'] = df['pickup_time'].dt.date
    rides_per_day = df.groupby('date').size()
    avg_fare = df['fare'].mean()

    driver_df = pd.read_csv('driver_details.csv')

    map_ = folium.Map(location=[df['pickup_lat'].mean(), df['pickup_lon'].mean()], zoom_start=12)
    for _, row in df.iterrows():
        folium.CircleMarker([row['pickup_lat'], row['pickup_lon']], radius=3, color='blue').add_to(map_)

    map_.save('static/heatmap.html')

    chart_labels = rides_per_day.index.astype(str).tolist()
    chart_values = rides_per_day.values.tolist()

    try:
        df_perf = pd.read_csv('ride_assignments.csv')
        y_true = df_perf['actual_driver_id'].astype(str)
        y_pred = df_perf['predicted_driver_id'].astype(str)

        accuracy = f"{round(accuracy_score(y_true, y_pred) * 100, 2)}%"
        precision = f"{round(precision_score(y_true, y_pred, average='macro', zero_division=0) * 100, 2)}%"
        recall = f"{round(recall_score(y_true, y_pred, average='macro', zero_division=0) * 100, 2)}%"
        f1 = f"{round(f1_score(y_true, y_pred, average='macro', zero_division=0) * 100, 2)}%"

    except Exception as e:
        accuracy = precision = recall = f1 = 'N/A'

    return render_template(
        'admin.html',
        rides_per_day=rides_per_day,
        avg_fare=round(avg_fare, 2),
        chart_labels=chart_labels,
        chart_values=chart_values,
        uber_headers=df.columns.tolist(),
        uber_data=df.head(5).values.tolist(),
        driver_headers=driver_df.columns.tolist(),
        driver_data=driver_df.head(5).values.tolist(),
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1
    )
    
if __name__ == '__main__':
    app.run(debug=True)