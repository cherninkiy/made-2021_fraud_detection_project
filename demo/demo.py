import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import folium_static

from utils import *


### Streamlit page config

st.set_page_config(page_title="TrackViz")

st.markdown(""" <style> 
        #MainMenu {visibility: hidden;} 
        footer {visibility: hidden;} 
        </style> """, unsafe_allow_html=True)

padding = 0

st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

creds = make_creds()

if "project_folders" not in st.session_state.keys():
    st.session_state['project_folders'] = {
        "Fraud": "12rmMvI9YfS1eF-KXmoftQjTxo4JigZgB",
        "Good": "1gZnyMi7qKkkaN4STQeKRiDz5T8iLa5sg",
        "Unlabeled" : "159iWBL-UTM6_bvULnOMjonor6SACq6io"
    }
project_folders = st.session_state["project_folders"]
print("Gdrive folders:")
print("\n".join([f"\t{k}: {v}" for k, v in project_folders.items()]))

if "drivers_files" not in st.session_state.keys():
    drivers_files = pd.DataFrame([], columns=["id", "name", "fraud"])
    for k in project_folders.keys():
        print(f"Load {k} drivers files")
        files = get_gdrive_file_list(project_folders[k], credentials=creds)
        df = pd.DataFrame(files, columns=("id", "name"))
        df = df[~df["name"].str.startswith(".")].copy()
        df["fraud"] = k
        drivers_files = pd.concat((drivers_files, df), axis=0)
    drivers_files.columns = ["gdrive_id", "driver_hash", "fraud"]
    st.session_state["drivers_files"] = drivers_files
drivers_files = st.session_state["drivers_files"]
print(f"Loaded {drivers_files.shape[0]} drivers")

if "result_preds" not in st.session_state.keys():
    result_csv = load_gdrive_file_data("1r3wFu7U30ozspUe-wCEXYUzWXnXUUNLu", credentials=creds)
    st.session_state['result_preds'] = pd.read_csv(result_csv)
result_preds = st.session_state["result_preds"]

def refresh():
    if "folder_id" in st.session_state.keys():
        del st.session_state["folder_id"]
    if "driver_hash" in st.session_state.keys():
        del st.session_state["driver_hash"]
    if "driver_fraud" in st.session_state.keys():
        del st.session_state["driver_fraud"]
    if "df_driver_gps" in st.session_state.keys():
        del st.session_state["df_driver_gps"]
    if "df_driver_accel" in st.session_state.keys():
        del st.session_state["df_driver_accel"]

st.sidebar.markdown("# Track Vizualizer")
fraud_pick = st.sidebar.radio("Driver category", ("Random", "Fraud", "Good", "Unlabeled"), on_change=refresh)
map_gps = st.sidebar.checkbox("Map GPS data", value=True)
map_accel = st.sidebar.checkbox("Map acceleration data", value=True)
st.sidebar.button("Refresh", on_click=refresh)

if not "driver_hash" in st.session_state.keys():
    if fraud_pick != "Random":
        mask = drivers_files["fraud"] == fraud_pick
        driver_hash = np.random.choice(drivers_files.loc[mask, "driver_hash"].unique())
    else:
        driver_hash = np.random.choice(drivers_files["driver_hash"].unique())
    mask = drivers_files["driver_hash"] == driver_hash
    folder_id, driver_hash, driver_fraud = drivers_files[mask].values[0]
    st.session_state["folder_id"] = folder_id
    st.session_state["driver_hash"] = driver_hash
    st.session_state["driver_fraud"] = driver_fraud
    print("Sampled driver:")
else:
    print("Cached driver:")

folder_id = st.session_state["folder_id"]
driver_hash = st.session_state["driver_hash"]
driver_fraud = st.session_state["driver_fraud"]
print(f"\tfolder_id: {folder_id}")
print(f"\tdriver_hash: {driver_hash}")
print(f"\tdriver_fraud: {driver_fraud}")

st.write("Driver category:", driver_fraud)
st.write("Driver hash:", driver_hash)
st.write("Predictions:")
mask = result_preds["driver_hash"] == int(driver_hash)
st.write(result_preds[mask].drop(columns="driver_hash"))

if map_gps or map_accel:
    files = get_gdrive_file_list(folder_id, credentials=creds)
    paths = pd.DataFrame(files)

    if "df_driver_gps" in st.session_state.keys():
        df_driver_gps = st.session_state["df_driver_gps"]
    else:
        df_driver_gps = pd.DataFrame((), columns=["time", "lat", "lon", "gps_time"])
        if map_gps:
            gps_files = paths.loc[paths["name"] == "track.csv", "id"]
            if gps_files.shape[0] > 0:
                gps_csv = load_gdrive_file_data(gps_files.values[0], credentials=creds)
                df_driver_gps = pd.read_csv(gps_csv, parse_dates=[0, 3])
                df_driver_gps["driver_hash"] = driver_hash
                df_driver_gps["fraud"] = {"Good": 0, "Fraud": 1, "Unlabeled": -1}[driver_fraud]
                df_driver_gps = df_driver_gps.sort_values(by="gps_time")
        st.session_state["df_driver_gps"] = df_driver_gps

    if "df_driver_accel" in st.session_state.keys():
        df_driver_accel = st.session_state["st.session_state"]
    else:
        df_driver_accel = pd.DataFrame((), columns=["time", "x", "y", "z", "lat", "lon"])
        if map_accel:
            accel_files = paths.loc[paths["name"] == "accelerometer.csv", "id"]
            if accel_files.shape[0] > 0:
                accel_csv = load_gdrive_file_data(accel_files.values[0], credentials=creds)
                df_driver_accel = pd.read_csv(accel_csv, parse_dates=[0])
                df_driver_accel["driver_hash"] = driver_hash
                df_driver_accel["fraud"] = {"Good": 0, "Fraud": 1, "Unlabeled": -1}[driver_fraud]
                df_driver_accel["time"] = df_driver_accel["time"].dt.tz_convert(None) + pd.DateOffset(hours=3)
                if driver_fraud == "Good":
                    df_driver_accel["time"] = df_driver_accel["time"] + pd.DateOffset(hours=-24)
                df_driver_accel = df_driver_accel.sort_values(by="time")
        st.session_state["df_driver_gps"] = df_driver_gps

    center = (37.6261, 55.7532)
    if df_driver_gps.shape[0] > 0:
        center = df_driver_gps[["lat", "lon"]].mean()
    if df_driver_accel.shape[0] > 0:
        center = df_driver_accel[["lat", "lon"]].mean()

    m = folium.Map(location=center, zoom_start=10)

    if map_gps:
        label = f"gps ({df_driver_gps.shape[0]} points)"
        map_driver_points(m, df_driver_gps, label=label, color="purple")
    if map_accel:
        label = f"accel ({df_driver_accel.shape[0]} points)"
        map_driver_points(m, df_driver_accel, label=label)

    folium.map.LayerControl('bottomleft', collapsed=False).add_to(m)
    folium_static(m)

