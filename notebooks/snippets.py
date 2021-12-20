import pandas as pd
import numpy as np
import folium
from math import radians, cos, sin, asin, acos, sqrt
from matplotlib import pyplot as plt
from matplotlib import colors
from IPython.display import display




def haversine_approx(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # haversine formula 
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    meters = km * 1000
    return meters


def angle_approx(lat1_v1, lon1_v1, lat2_v1, lon2_v1, lat1_v2, lon1_v2, lat2_v2, lon2_v2, tol=1e-9):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'
    https://stackoverflow.com/questions/3380628/fast-arc-cos-algorithm
    """
    x1, y1 = (lat2_v1 - lat1_v1, lon2_v1 - lon1_v1)
    x2, y2 = (lat2_v2 - lat1_v2, lon2_v2 - lon1_v2)
    norm_v1 = sqrt(x1**2 + y1**2)
    if norm_v1 < tol:
        return 0.0
    norm_v2 = sqrt(x2**2 + y2**2)
    if norm_v2 < tol:
        return 0.0
    # from 0 to pi
    angle1 = acos(np.clip(x1 / norm_v1, -1.0, 1.0))
    angle2 = acos(np.clip(x2 / norm_v2, -1.0, 1.0))
    # from -pi to pi
    angle1 = angle1 * (y1 > 0) - angle1 * (y1 < 0)
    angle2 = angle2 * (y2 > 0) - angle2 * (y2 < 0)
    return angle1 - angle2


def describe(df : pd.DataFrame):
    """
    Describe pandas dataframe
    """
    display(pd.concat((pd.Series(df.dtypes, name="dtypes"), df.describe(datetime_is_numeric=True).T), axis=1))
    display(df.shape)
    

def sample_driver(df, force_fraud, col_fraud="fraud"):
    """
    Sample random driver from DataFrame
    """
    if not force_fraud is None:
        driver_hash = np.random.choice(df.loc[df[col_fraud] == force_fraud, "driver_hash"].unique())
    else:
        driver_hash = np.random.choice(df["driver_hash"].unique())
    driver_data = df[df["driver_hash"] == driver_hash]
    driver_fraud = driver_data[col_fraud].replace({-1:"Unknown", 0:"Good", 1:"Fraud"}).values[0]
    return driver_data, driver_hash, driver_fraud
    
    
def plot_fraud_hist(df, column, ylog=True):
    """
    Plot column histogramm
    """
    hist = df[df["fraud"] == -1][column].value_counts()
    plt.bar(x=hist.index, height=hist.values, alpha=0.8, color="blue", label="Unknown")
    xlim1 = hist.index.max()
    hist = df[df["fraud"] == 0][column].value_counts()
    plt.bar(x=hist.index, height=hist.values, alpha=0.8, color="green", label="Good")
    xlim2 = hist.index.max()
    hist = df[df["fraud"] == 1][column].value_counts()
    plt.bar(x=hist.index, height=hist.values, alpha=0.8, color="red", label="Fraud")
    xlim3 = hist.index.max()
    plt.xlim((None, np.amax((xlim1, xlim2, xlim3))))
    if ylog:
        plt.yscale("log")
    plt.legend()
    plt.title(column)
    

def plot_fraud_pca(df, feats, driver_hash, title=""):
    mask = df["fraud"] == -1
    plt.scatter(feats[mask,0], feats[mask,1], color="blue", label="Other", s=10)
    mask = df["fraud"] == 0
    plt.scatter(feats[mask,0], feats[mask,1], color="green", label="Good", s=100, marker="*")
    mask = df["fraud"] == 1
    plt.scatter(feats[mask,0], feats[mask,1], color="red", label="Fraud", s=50, marker="+")
    mask = df["driver_hash"] == driver_hash
    plt.scatter(feats[mask,0], feats[mask,1], color="yellow", label=driver_hash)
    plt.legend(loc='upper right')
    plt.title(title)
    
    
def map_driver_points(m, df, driver_hash, label="", color=None, weight=6):
    cmap = list(colors.XKCD_COLORS.values())
    if color is None:
        color = df.head(1)["fraud"].replace({1:"red", 0:"green", -1:"yellow"}).values[0]
    elif color == "random":
        color = cmap[np.random.choice(len(cmap))]
    name = f'<span style="color: {color};">{driver_hash} [{label}]</span>'
    fg = folium.FeatureGroup(name).add_to(m)
    folium.PolyLine(df[["lat", "lon"]], color=color, weight=weight, opacity=0.8).add_to(fg)
    return m


def map_driver_routs(m, df_driver, route_idx, label, min_points=10):
    cmap = list(colors.XKCD_COLORS.values())
    legend = '<span style="color: {col};">{label}_{txt}_({num})</span>'
    
    driver_routs = df_driver.loc[route_idx.index, ["lat", "lon"]]
    for i, idx in enumerate(route_idx.value_counts().index):
        df_route = driver_routs.loc[route_idx[route_idx == idx].index,:]
        num_points = df_route.shape[0]
        if df_route.shape[0] < min_points:
            continue
        color = cmap[np.random.choice(len(cmap))]
        name = legend.format(col=color, label=label, txt=idx, num=num_points)
        fg = folium.FeatureGroup(name).add_to(m)
        folium.PolyLine(df_route, color=color, weight=8, opacity=0.8).add_to(fg) 