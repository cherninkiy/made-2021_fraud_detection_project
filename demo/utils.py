import os.path
import pickle
from io import BytesIO
import numpy as np
import pandas as pd
import folium
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib import colors

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload


def make_creds():
    return service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )

def get_gdrive_file_list(folder_id, credentials):
    service = build("drive", "v3", credentials=credentials)
    result = service.files() \
        .list(
            q=f'"{folder_id}" in parents',
            pageSize=400,
            fields="files(id, name)"
        ) \
        .execute()
    files = result.get("files")
    return files


def load_gdrive_file_data(file_id, credentials):
    service = build("drive", "v3", credentials=credentials)
    request = service.files().get_media(fileId=file_id)
    buffer = BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        print("Download %d%%." % int(status.progress() * 100))
    if done:
        buffer.seek(0)
        return buffer
    return None


def map_driver_points(m, df, label="", color=None, weight=6):
    cmap = list(colors.XKCD_COLORS.values())
    if color is None:
        color = df.head(1)["fraud"].replace({1:"red", 0:"green", -1:"blue"}).values[0]
    elif color == "random":
        color = cmap[np.random.choice(len(cmap))]
    name = f'<span style="color: {color};">{label}</span>'
    fg = folium.FeatureGroup(name).add_to(m)
    folium.PolyLine(df[["lat", "lon"]], color=color, weight=weight, opacity=0.8).add_to(fg)
    return m