import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
from geoclip import GeoCLIP
import requests
import sys
from PIL import Image
import piexif
import psycopg2
import base64
import cv2
import argparse
import io
import json
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as sch
from collections import defaultdict, Counter
import networkx as nx
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer
import re
import pandas as pd
import matplotlib as mpl
import geopandas as gpd
import shapely.geometry as shp_geom
from datashader.bundling import hammer_bundle
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import ScalarFormatter
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderQuotaExceeded
import time
from pytube import YouTube
import validators
from sentence_transformers import SentenceTransformer
import bs4
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.linalg
import matplotlib.patches as mpl_patches
from wordcloud import WordCloud
import pandas as pd

# Database connection (updated to match Node.js local DB)
connection = psycopg2.connect(
    host="127.0.0.1",
    database="postgres",
    user="postgres",
    password="postgres",
    port="5432"
)
cursor = connection.cursor()
def convert_to_serializable(obj):
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist() if isinstance(obj, np.ndarray) else obj.item()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj

def fix_entity_indices(project):
    try:
        select_query = f'SELECT id, content FROM "{project}_nodes";'
        cursor.execute(select_query)
        rows = cursor.fetchall()
        for node_id, content in rows:
            updated = False
            entities_parent = content
            # Find entities within profileData
            if 'profileData' in content:
                for key in content['profileData']:
                    if 'entities' in content['profileData'][key]:
                        entities_parent = content['profileData'][key]
                        break
            if 'entities' in entities_parent:
                entities = entities_parent['entities']
                # Reassign indices starting from 1
                for idx, entity in enumerate(entities, 1):
                    if entity.get('index') != idx:
                        entity['index'] = idx
                        updated = True
                        print(f"Updated index for entity in node {node_id}: {entity.get('name', entity.get('alt', 'Unknown'))} to {idx}")
                if updated:
                    update_query = f'UPDATE "{project}_nodes" SET content = %s WHERE id = %s;'
                    cursor.execute(update_query, (json.dumps(content), node_id))
                    connection.commit()
                    print(f"Updated entity indices for node {node_id}")
    except Exception as e:
        print(f"Error fixing entity indices: {e}")
        connection.rollback()
def download_all_images(project, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    query = f"""SELECT id, content, type FROM "{project}_nodes";"""
    cursor.execute(query)
    rows = cursor.fetchall()
    downloaded_paths = []
    for node_id, content, node_type in rows:
        entities = []
        if 'entities' in content:
            entities = content['entities']
        elif 'profileData' in content:
            for key in content['profileData']:
                if 'entities' in content['profileData'][key]:
                    entities = content['profileData'][key]['entities']
                    break
        elif 'image' in content:
            entities = [{'type': 'person', 'image': content['image'], 'details': content.get('details', {}) }]
        for idx, entity in enumerate(entities):
            entity_idx = entity.get("index", idx + 1)
            img_src = None
            img_base64 = None
            if entity['type'] == 'person' and 'image' in entity and entity['image']:
                image_obj = entity['image']
                if isinstance(image_obj, dict):
                    img_src = image_obj.get('src')
                    img_base64 = image_obj.get('base64')
                elif isinstance(image_obj, str):
                    img_src = image_obj
            elif entity['type'] == 'image':
                img_src = entity.get('src')
                img_base64 = entity.get('base64')
            img_path = os.path.join(output_folder, f"node_{node_id}_entity_{entity_idx}.jpg")
            if os.path.exists(img_path):
                print(f"Image already downloaded: {img_path}")
                downloaded_paths.append(img_path)
                continue
            img_data = None
            if img_src:
                if img_src.startswith('data:'):
                    if 'svg' in img_src.lower():
                        print(f"Skipping SVG data URL for node {node_id}, entity {entity_idx}")
                        continue
                    try:
                        _, base64_data = img_src.split(',', 1)
                        img_data = base64.b64decode(base64_data)
                    except Exception as e:
                        print(f"Error decoding data URL as base64: {e}")
                        continue
                elif img_src.startswith('//'):
                    img_src = 'https:' + img_src
                    try:
                        response = requests.get(img_src)
                        if response.status_code == 200:
                            img_data = response.content
                    except Exception as e:
                        print(f"Error downloading image from {img_src}: {e}")
                elif img_src.startswith('/'):
                    print(f"Skipping relative URL: {img_src}")
                    continue
                else:
                    try:
                        response = requests.get(img_src)
                        if response.status_code == 200:
                            img_data = response.content
                    except Exception as e:
                        print(f"Error downloading image from {img_src}: {e}")
                        # Fallback to base64 decode
                        try:
                            img_data = base64.b64decode(img_src)
                        except Exception as base64_e:
                            print(f"Error decoding img_src as base64: {base64_e}")
            elif img_base64:
                try:
                    img_data = base64.b64decode(img_base64)
                except Exception as e:
                    print(f"Error decoding base64 image: {e}")
            if img_data:
                try:
                    Image.open(io.BytesIO(img_data)).verify()
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    downloaded_paths.append(img_path)
                    print(f"Downloaded/saved image to {img_path}")
                except Exception as e:
                    print(f"Invalid image data for node {node_id}, entity {entity_idx}: {e}")
    return downloaded_paths
def download_all_videos(project, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    query = f"""SELECT id, content, type FROM "{project}_nodes";"""
    cursor.execute(query)
    rows = cursor.fetchall()
    downloaded_paths = []
    base_url = ""
    for node_id, content, node_type in rows:
        entities = []
        if 'entities' in content:
            entities = content['entities']
        elif 'profileData' in content:
            for key in content['profileData']:
                if 'entities' in content['profileData'][key]:
                    entities = content['profileData'][key]['entities']
                    break
        for idx, entity in enumerate(entities):
            entity_idx = entity.get("index", idx + 1)
            video_src = entity.get('src')
            video_base64 = entity.get('base64')
            video_embed = entity.get('embed')
            alt_text = entity.get('alt', 'No description provided')
            video_path = os.path.join(output_folder, f"node_{node_id}_entity_{entity_idx}.mp4")
            if os.path.exists(video_path):
                print(f"Video already downloaded: {video_path}")
                downloaded_paths.append(video_path)
                continue
            video_data = None
            if video_src:
                # Handle relative URLs
                if video_src.startswith('/'):
                    video_src = base_url.rstrip('/') + video_src
                # Handle protocol-relative URLs
                elif video_src.startswith('//'):
                    video_src = 'https:' + video_src
                # Validate URL
                if validators.url(video_src):
                    try:
                        response = requests.get(video_src, stream=True, timeout=10)
                        if response.status_code == 200:
                            video_data = response.content
                            print(f"Downloaded video from src: {video_src}")
                        else:
                            print(f"Failed to download video from {video_src}: Status {response.status_code}")
                    except Exception as e:
                        print(f"Error downloading video from {video_src}: {e}")
                else:
                    print(f"Invalid video src URL: {video_src}")
            elif video_base64:
                try:
                    video_data = base64.b64decode(video_base64)
                    print(f"Decoded base64 video for node {node_id}, entity {entity_idx}")
                except Exception as e:
                    print(f"Error decoding base64 video: {e}")
            elif video_embed:
                # Handle embedded videos (e.g., YouTube)
                if validators.url(video_embed):
                    try:
                        if 'youtube.com' in video_embed or 'youtu.be' in video_embed:
                            yt = YouTube(video_embed)
                            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                            if stream:
                                stream.download(output_path=output_folder, filename=f"node_{node_id}_entity_{entity_idx}.mp4")
                                video_path = os.path.join(output_folder, f"node_{node_id}_entity_{entity_idx}.mp4")
                                print(f"Downloaded YouTube video from embed: {video_embed}")
                                downloaded_paths.append(video_path)
                                continue # Skip further processing as file is already saved
                            else:
                                print(f"No suitable stream found for YouTube video: {video_embed}")
                        else:
                            print(f"Unsupported embed URL: {video_embed}")
                    except Exception as e:
                        print(f"Error downloading embedded video from {video_embed}: {e}")
                else:
                    print(f"Invalid embed URL: {video_embed}")
            if video_data:
                # Verify video data (basic check using OpenCV)
                try:
                    with open(video_path, 'wb') as f:
                        f.write(video_data)
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        print(f"Invalid video file saved: {video_path}")
                        os.remove(video_path)
                        continue
                    cap.release()
                    downloaded_paths.append(video_path)
                    print(f"Saved video to {video_path}")
                    # Update entity with alt text and verified src
                    entity['alt'] = alt_text
                    if video_src:
                        entity['src'] = video_src
                    elif video_embed:
                        entity['embed'] = video_embed
                except Exception as e:
                    print(f"Error saving/verifying video for node {node_id}, entity {entity_idx}: {e}")
    return downloaded_paths
def split_videos(video_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(video_folder):
        if not filename.lower().endswith('.mp4'):
            continue
        video_path = os.path.join(video_folder, filename)
        # Parse filename to extract node_id and entity_idx
        if not (filename.startswith("node_") and "_entity_" in filename):
            print(f"Skipping video with unexpected filename: {filename}")
            continue
        parts = filename.split("_")
        if len(parts) < 4:
            print(f"Skipping video with invalid filename format: {filename}")
            continue
        node_id = parts[1]
        entity_idx = parts[3].split(".")[0] # remove .mp4
        # Check if frames already exist
        first_frame = os.path.join(output_folder, f"node_{node_id}_entity_{entity_idx}_frame_0.jpg")
        if os.path.exists(first_frame):
            print(f"Frames already exist for {filename}, skipping split.")
            continue
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 10) if fps > 0 else 300 # fallback
        count = 0
        frame_idx = 0
        success, frame = cap.read()
        while success:
            if count % frame_interval == 0:
                frame_path = os.path.join(output_folder, f"node_{node_id}_entity_{entity_idx}_frame_{frame_idx}.jpg")
                cv2.imwrite(frame_path, frame)
                print(f"Saved frame: {frame_path}")
                frame_idx += 1
            success, frame = cap.read()
            count += 1
        cap.release()
def get_image_filename(image_field):

    if isinstance(image_field, str):
        return image_field
    elif isinstance(image_field, dict):
        # Adjust this based on the actual structure of the dictionary
        # Common keys to check: 'url', 'filename', 'path'
        for key in ['url', 'filename', 'path']:
            if key in image_field and isinstance(image_field[key], str):
                return image_field[key]
    return None
# Load YOLO model for object detection
def load_detection_model():
    model_path = "yolov8n.pt" # Pre-trained YOLOv8 small model
    if not os.path.exists(model_path):
        print("Downloading YOLO model...")
        torch.hub.download_url_to_file(
            'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt', model_path
        )
    return YOLO(model_path)
def detect_objects(image_path, model):
    """Detect objects in the image using the YOLO model."""
    results = model(image_path) # Perform detection
    detections = []
    for result in results: # Iterate over detected objects
        for box in result.boxes: # Access bounding boxes
            class_id = int(box.cls.item()) # Convert class tensor to an integer
            label = result.names[class_id] # Class name
            confidence = float(box.conf.item()) # Confidence score
            detections.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": box.xyxy.tolist() # Boundingbox coordinates
                })
    return detections
def extract_metadata(image_path):
    try:
        img = Image.open(image_path)
        exif_bytes = img.info.get("exif")
        if exif_bytes is None or len(exif_bytes) == 0:
            exif_data = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        else:
            exif_data = piexif.load(exif_bytes)
        metadata = {}
        if "GPS" in exif_data and exif_data["GPS"]:
            gps_info = exif_data["GPS"]
            def convert_to_degrees(value):
                d, m, s = value
                return d + (m / 60.0) + (s / 3600.0)
            lat_ref = gps_info.get(piexif.GPSIFD.GPSLatitudeRef, b"").decode("utf-8")
            lon_ref = gps_info.get(piexif.GPSIFD.GPSLongitudeRef, b"").decode("utf-8")
            lat = convert_to_degrees(gps_info[piexif.GPSIFD.GPSLatitude])
            lon = convert_to_degrees(gps_info[piexif.GPSIFD.GPSLongitude])
            if lat_ref == "S":
                lat = -lat
            if lon_ref == "W":
                lon = -lon
            metadata['lat'] = lat
            metadata['lon'] = lon
        # Extract timestamp
        timestamp = None
        if "Exif" in exif_data:
            date_time_original = exif_data["Exif"].get(piexif.ExifIFD.DateTimeOriginal)
            if date_time_original:
                timestamp = date_time_original.decode("utf-8")
        if not timestamp and "0th" in exif_data:
            date_time = exif_data["0th"].get(piexif.ImageIFD.DateTime)
            if date_time:
                timestamp = date_time.decode("utf-8")
        if timestamp:
            metadata['timestamp'] = timestamp
        return metadata
    except Exception as e:
        print(f"Error extracting metadata: {e}")
    return {}
def geocode_location(location_str, user_agent="my_analysis_app"):
    try:
        geolocator = Nominatim(user_agent=user_agent)
        location = geolocator.geocode(location_str, timeout=10)
        if location:
            return {
                "lat": location.latitude,
                "lon": location.longitude,
                "confidence": 3.0
            }
        else:
            print(f"Could not geocode location: {location_str}")
            return None
    except (GeocoderTimedOut, GeocoderQuotaExceeded) as e:
        print(f"Geocoding error for {location_str}: {e}")
        time.sleep(2)
        return None
    except Exception as e:
        print(f"Unexpected error geocoding {location_str}: {e}")
        return None
def process_image(image_path, endpoint_url, output_file, model, additional_info, project):
    # Check if image has already been processed
    filename = os.path.basename(image_path)
    parts = filename[:-4].split('_') # assume .jpg
    if len(parts) >= 4 and parts[0] == 'node' and parts[2] == 'entity':
        node_id = parts[1]
        entity_idx = int(parts[3])
    else:
        print(f"Skipping invalid filename for geo: {filename}")
        return

    file_mtime = os.path.getmtime(image_path)
    is_frame = len(parts) > 4 and parts[4] == 'frame'
    frame_idx = int(parts[5]) if is_frame and len(parts) > 5 else None

    # Variables to track if already processed and existing geo data
    already_processed = False
    existing_geo_data = None
    entity = None
    content = None

    try:
        select_query = f'SELECT content FROM "{project}_nodes" WHERE id = %s;'
        cursor.execute(select_query, (node_id,))
        row = cursor.fetchone()
        if not row:
            print(f"No node found for id {node_id}")
            return
        content = row[0]

        # Get entity
        entities_parent = content
        if 'profileData' in content:
            for key in content['profileData']:
                if 'entities' in content['profileData'][key]:
                    entities_parent = content['profileData'][key]
                    break

        if 'entities' in entities_parent:
            for ent in entities_parent['entities']:
                if ent.get('index') == entity_idx:
                    entity = ent
                    break

        if not entity and not ('image' in content and entity_idx == 1):
            print(f"No entity found for index {entity_idx} in node {node_id}")
            return

        # Check for existing geolocation and file modification time
        if 'image' in content and entity_idx == 1 and not is_frame:
            if 'geolocation' in content and 'lat' in content['geolocation'] and content.get('last_geo_processed', 0) >= file_mtime:
                print(f"Already processed image: {filename} (geolocation exists and file unchanged in content)")
                already_processed = True
                existing_geo_data = content['geolocation']
        elif entity:
            if is_frame:
                if 'frame_geolocation' in entity and frame_idx in entity['frame_geolocation'] and 'lat' in entity['frame_geolocation'][frame_idx] and entity.get('frame_last_geo_processed', {}).get(frame_idx, 0) >= file_mtime:
                    print(f"Already processed frame: {filename} (geolocation exists and file unchanged for frame {frame_idx})")
                    already_processed = True
                    existing_geo_data = entity['frame_geolocation'][frame_idx]
            elif 'geolocation' in entity and 'lat' in entity['geolocation'] and entity.get('last_geo_processed', 0) >= file_mtime:
                print(f"Already processed image: {filename} (geolocation exists and file unchanged for entity {entity_idx})")
                already_processed = True
                existing_geo_data = entity['geolocation']
    except Exception as e:
        print(f"Error checking existing geolocation for {filename}: {e}")
        return

    # If already processed, use existing data
    if already_processed and existing_geo_data:
        lat = existing_geo_data['lat']
        lon = existing_geo_data['lon']
        prob = existing_geo_data.get('probability', 1.0)
        source = existing_geo_data.get('source', 'existing')
        timestamp = existing_geo_data.get('timestamp', None)
        print(f"Using existing geolocation data for {filename}: lat={lat:.6f}, lon={lon:.6f}")
    else:
        # Process the image to get geolocation
        metadata = extract_metadata(image_path)
        timestamp = metadata.get('timestamp')
        lat = lon = prob = None
        source = "metadata"

        if 'lat' in metadata and 'lon' in metadata:
            lat = metadata['lat']
            lon = metadata['lon']
            prob = 4.0
            print(f"Metadata Geolocation Found: Latitude={lat:.6f}, Longitude={lon:.6f}")
        elif entity and 'details' in entity and 'location' in entity['details'] and entity['details']['location']:
            location_str = entity['details']['location']
            if 'geolocation' in entity and 'lat' in entity['geolocation']:
                lat = entity['geolocation']['lat']
                lon = entity['geolocation']['lon']
                prob = entity['geolocation'].get('probability', 3.5)
                source = "entity_existing"
                print(f"Using existing entity geolocation for {filename}: ({lat}, {lon})")
            else:
                geodata = geocode_location(location_str)
                if geodata:
                    lat = geodata['lat']
                    lon = geodata['lon']
                    prob = geodata['confidence']
                    source = "entity_geocoded"
                    entity['geolocation'] = {"lat": lat, "lon": lon, "probability": prob}
                    update_query = f'UPDATE "{project}_nodes" SET content = %s WHERE id = %s;'
                    cursor.execute(update_query, (json.dumps(content), node_id))
                    connection.commit()
                    print(f"Geocoded and updated entity location for {filename}: {location_str} -> ({lat}, {lon})")
                    time.sleep(1)
        elif entity and 'nlp_analysis' in entity:
            nlp = entity['nlp_analysis']
            if 'named_entities' in nlp:
                locs = [e for e in nlp['named_entities'] if e['label'] == 'LOC' and e['score'] > 0.5]
                if locs:
                    locs.sort(key=lambda x: x['score'], reverse=True)
                    location_str = locs[0]['text']
                    geodata = geocode_location(location_str)
                    if geodata:
                        lat = geodata['lat']
                        lon = geodata['lon']
                        prob = 2.0  # Set probability for NLP geocoded
                        source = "nlp_geocoded"
                        entity['geolocation'] = {"lat": lat, "lon": lon, "probability": prob, "source": source}
                        update_query = f'UPDATE "{project}_nodes" SET content = %s WHERE id = %s;'
                        cursor.execute(update_query, (json.dumps(content), node_id))
                        connection.commit()
                        print(f"Geocoded NLP location for {filename}: {location_str} -> ({lat}, {lon})")
                        time.sleep(1)
                else:
                    print(f"No locations found with score > 0.5 for {filename}")
            else:
                print(f"Error: 'named_entities' not found in nlp_analysis for {filename}. NLP data: {nlp}")

        if lat is None or lon is None:
            print("No metadata or entity location found. Running GeoCLIP.")
            try:
                from PIL import Image
                img = Image.open(image_path)
                width, height = img.size
                if width < 16 or height < 16:
                    print(f"Skipping too small image: {filename} (size: {width}x{height})")
                    return
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
                return

            top_pred_gps, top_pred_prob = model.predict(image_path, top_k=1)
            if len(top_pred_gps) == 0:
                print(f"No GPS coordinates found for: {image_path}")
                return
            lat_tensor, lon_tensor = top_pred_gps[0]
            prob_tensor = top_pred_prob[0]
            lat = lat_tensor.item()
            lon = lon_tensor.item()
            prob = prob_tensor.item()
            source = "geoclip"

        # Ensure we have valid coordinates before proceeding
        if lat is None or lon is None:
            print(f"Failed to extract coordinates for: {image_path}")
            return

        # Ensure prob has a value based on source
        if prob is None:
            if source == "metadata":
                prob = 4.0
            elif source == "entity_existing":
                prob = 3.5
            elif source == "entity_geocoded":
                prob = 3.0
            elif source == "nlp_geocoded":
                prob = 2.0
            elif source == "geoclip":
                prob = 1.0
            else:
                prob = 1.0  # Default fallback

        print(f"Extracted Coordinates from {source}: Latitude={lat:.6f}, Longitude={lon:.6f}, Probability={prob:.6f}")
        if timestamp:
            print(f"Extracted Timestamp: {timestamp}")

        # Update DB with geolocation (only if not already processed)
        try:
            select_query = f'SELECT content FROM "{project}_nodes" WHERE id = %s;'
            cursor.execute(select_query, (node_id,))
            row = cursor.fetchone()
            if row:
                content = row[0]
                found = False
                if 'image' in content and entity_idx == 1 and not is_frame:
                    content['geolocation'] = {"lat": lat, "lon": lon, "probability": prob, "source": source}
                    if timestamp:
                        content['geolocation']['timestamp'] = timestamp
                    content['last_geo_processed'] = file_mtime
                    found = True
                else:
                    entities_parent = content
                    if 'profileData' in content:
                        for key in content['profileData']:
                            if 'entities' in content['profileData'][key]:
                                entities_parent = content['profileData'][key]
                                break
                    if 'entities' in entities_parent:
                        for ent in entities_parent['entities']:
                            if ent.get('index') == entity_idx:
                                if is_frame:
                                    if 'frame_geolocation' not in ent:
                                        ent['frame_geolocation'] = {}
                                    ent['frame_geolocation'][frame_idx] = {"lat": lat, "lon": lon, "probability": prob, "source": source}
                                    if timestamp:
                                        ent['frame_geolocation'][frame_idx]['timestamp'] = timestamp
                                    if 'frame_last_geo_processed' not in ent:
                                        ent['frame_last_geo_processed'] = {}
                                    ent['frame_last_geo_processed'][frame_idx] = file_mtime
                                else:
                                    ent['geolocation'] = {"lat": lat, "lon": lon, "probability": prob, "source": source}
                                    if timestamp:
                                        ent['geolocation']['timestamp'] = timestamp
                                    ent['last_geo_processed'] = file_mtime
                                found = True
                                break
                if found:
                    update_query = f'UPDATE "{project}_nodes" SET content = %s WHERE id = %s;'
                    cursor.execute(update_query, (json.dumps(content), node_id))
                    connection.commit()
                    print(f"Updated geolocation for node {node_id}, entity {entity_idx}" + (f", frame {frame_idx}" if is_frame else ""))
                else:
                    print(f"No place to add geolocation for node {node_id}, entity {entity_idx}")
        except Exception as e:
            print(f"Error updating geolocation for {filename}: {e}")

    # ALWAYS send the payload, whether already processed or not
    payload = {
        "lat": lat,
        "lon": lon,
        "probability": prob,
        "image": os.path.basename(image_path),
        "source": source,
        "node_id": node_id,
        "entity_idx": entity_idx,
        "entity": entity
    }
    if timestamp:
        payload["timestamp"] = timestamp

    try:
        response = requests.post(endpoint_url, json=payload)
        if response.status_code == 200:
            print(f"Coordinates sent successfully for image: {image_path}")
        else:
            print(f"Server returned status {response.status_code} for {image_path}: {response.text}")
    except Exception as e:
        print(f"Error sending coordinates for {image_path}: {e}")
def send_all_markers(endpoint_url, project):
    try:
        # Query to get all nodes with content
        select_query = f'SELECT id, content FROM "{project}_nodes";'
        cursor.execute(select_query)
        rows = cursor.fetchall()

        markers_sent = 0

        for row in rows:
            node_id = row[0]
            content = row[1]

            # Extract centrality values from the node
            node_centralities = content.get('centralities', {})
            node_person_centralities = content.get('person_centralities', {})

            # Check for geolocation in main content (entity_idx = 1)
            if 'image' in content and 'geolocation' in content:
                geo = content['geolocation']
                if 'lat' in geo and 'lon' in geo:
                    payload = {
                        "lat": geo['lat'],
                        "lon": geo['lon'],
                        "probability": geo.get('probability', 1.0),
                        "source": geo.get('source', 'existing'),
                        "node_id": node_id,
                        "entity_idx": 1,
                        "entity": None
                    }

                    # Add centrality values if they exist
                    if node_centralities:
                        payload["centralities"] = node_centralities
                    if node_person_centralities:
                        payload["person_centralities"] = node_person_centralities

                    if 'timestamp' in geo:
                        payload["timestamp"] = geo['timestamp']

                    # Send to endpoint
                    send_marker(endpoint_url, payload, f"node_{node_id}_entity_1")
                    markers_sent += 1

            # Check for entities with geolocation
            entities_parent = content
            if 'profileData' in content:
                for key in content['profileData']:
                    if 'entities' in content['profileData'][key]:
                        entities_parent = content['profileData'][key]
                        break

            if 'entities' in entities_parent:
                for entity in entities_parent['entities']:
                    entity_idx = entity.get('index')
                    if not entity_idx:
                        continue

                    # Create enhanced entity with centrality values
                    enhanced_entity = entity.copy()
                    if node_centralities:
                        enhanced_entity['centralities'] = node_centralities
                    if node_person_centralities:
                        enhanced_entity['person_centralities'] = node_person_centralities

                    # Check for regular geolocation
                    if 'geolocation' in entity:
                        geo = entity['geolocation']
                        if 'lat' in geo and 'lon' in geo:
                            payload = {
                                "lat": geo['lat'],
                                "lon": geo['lon'],
                                "probability": geo.get('probability', 1.0),
                                "source": geo.get('source', 'existing'),
                                "node_id": node_id,
                                "entity_idx": entity_idx,
                                "entity": enhanced_entity  # Use enhanced entity with centralities
                            }

                            # Add centrality values at the top level too
                            if node_centralities:
                                payload["centralities"] = node_centralities
                            if node_person_centralities:
                                payload["person_centralities"] = node_person_centralities

                            if 'timestamp' in geo:
                                payload["timestamp"] = geo['timestamp']

                            # Send to endpoint
                            send_marker(endpoint_url, payload, f"node_{node_id}_entity_{entity_idx}")
                            markers_sent += 1

                    # Check for frame geolocations
                    if 'frame_geolocation' in entity:
                        for frame_idx, frame_geo in entity['frame_geolocation'].items():
                            if 'lat' in frame_geo and 'lon' in frame_geo:
                                payload = {
                                    "lat": frame_geo['lat'],
                                    "lon": frame_geo['lon'],
                                    "probability": frame_geo.get('probability', 1.0),
                                    "source": frame_geo.get('source', 'existing'),
                                    "node_id": node_id,
                                    "entity_idx": entity_idx,
                                    "entity": enhanced_entity,  # Use enhanced entity with centralities
                                    "frame_idx": frame_idx
                                }

                                # Add centrality values at the top level too
                                if node_centralities:
                                    payload["centralities"] = node_centralities
                                if node_person_centralities:
                                    payload["person_centralities"] = node_person_centralities

                                if 'timestamp' in frame_geo:
                                    payload["timestamp"] = frame_geo['timestamp']

                                # Send to endpoint
                                send_marker(endpoint_url, payload, f"node_{node_id}_entity_{entity_idx}_frame_{frame_idx}")
                                markers_sent += 1

        print(f"\nTotal markers sent: {markers_sent}")

    except Exception as e:
        print(f"Error loading and sending markers: {e}")
        connection.rollback()


def send_marker(endpoint_url, payload, identifier):
    try:
        response = requests.post(endpoint_url, json=payload)
        if response.status_code == 200:
            print(f"Marker sent successfully: {identifier}")
        else:
            print(f"Server returned status {response.status_code} for {identifier}: {response.text}")
    except Exception as e:
        print(f"Error sending marker {identifier}: {e}")
def main():
    parser = argparse.ArgumentParser(description="Run analysis for a project.")
    parser.add_argument("project", help="The project name")
    parser.add_argument("--analyses", help="Comma-separated list of analyses to run")
    parser.add_argument("--face_image_file", help="Path to file containing base64-encoded face image for matching")
    args = parser.parse_args()

    project = args.project
    analyses_list = args.analyses.split(',') if args.analyses else []
    print(f"Running for project: {project}")
    print(f"Analyses: {analyses_list}")

    print(args.face_image_file);

    # Load base64 face image from file if provided
    uploaded_face_embedding = None

    print("\nFixing entity indices for all nodes...")
    fix_entity_indices(project)

    endpoint_url = 'http://127.0.0.1:4000/geoclip'
    project_dir = f"./temp/{project}"
    os.makedirs(project_dir, exist_ok=True)
    image_folder = os.path.join(project_dir, 'images')
    os.makedirs(image_folder, exist_ok=True)
    video_folder = os.path.join(project_dir, 'videos')
    os.makedirs(video_folder, exist_ok=True)
    output_folder = os.path.join(project_dir, 'output')
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'geo_data.txt')
    additional_info = "User-provided additional information"

    send_all_markers(endpoint_url, project)

    if 'download' in analyses_list:
        download_all_images(project, image_folder)
        download_all_videos(project, video_folder)
        split_videos(video_folder, image_folder)

    # Continue with face recognition, clustering, NLP, etc.
    if 'nlp' in analyses_list:
            print("\n=== Starting Extended NLP Analysis ===")
            # Initialize pipelines
            print("Loading NLP models...")
            # Check for GPU availability
            device = 0 if torch.cuda.is_available() else -1
            print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
            # Basic pipelines with explicit model and revision
            ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", revision="4c53496", aggregation_strategy="simple", device=device)
            sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", revision="714eb0f", device=device)
            language_pipeline = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection", device=device)
            zero_shot_pipeline = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
            translate_pipeline = pipeline("translation", model="facebook/nllb-200-distilled-600M", device=device)
            # Add emotion pipeline from second script
            emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, device=device)

            # Character-based chunking parameters
            char_chunk_size = 100000  # Approximate safe character length (~2000-3000 tokens)
            char_chunk_overlap = 200  # Overlap in characters
            # Define optional analyses - now including emotion
            enabled_analyses = ['language', 'sentiment', 'ner', 'emotion']
            # Storage for analysis results
            all_nlp_results = []
            sentiment_scores = []
            emotion_distributions = []  # Add emotion distributions storage
            topic_distributions = []
            entity_counts = defaultdict(int)
            all_words = []
            all_bigrams = []
            all_trigrams = []
            bigram_freq = Counter()  # Initialize bigram_freq here
            trigram_freq = Counter()  # Initialize trigram_freq here
            # Allow user to specify number of nodes to process
            max_nodes = None  # Default: process all nodes
            max_chunks = None  # Default: limit to 10 chunks per node, set to None to process all
            query = f"""SELECT id, content, page_text FROM "{project}_nodes" WHERE page_text IS NOT NULL AND page_text != '';"""
            cursor.execute(query)
            rows = cursor.fetchall()
            # Limit rows if max_nodes is specified
            if max_nodes is not None:
                rows = rows[:max_nodes]
            processed_nodes = 0
            for node_id, content, page_text in rows:
                full_text = page_text
                words = re.findall(r'\w+', full_text.lower())
                bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
                trigrams = [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)]
                all_words.extend(words)
                all_bigrams.extend(bigrams)
                all_trigrams.extend(trigrams)
                # Check if already processed
                already_processed = False
                entities_parent = content
                nlp_data = None
                if 'profileData' in content:
                    for key in content['profileData']:
                        if 'entities' in content['profileData'][key]:
                            entities_parent = content['profileData'][key]
                            break
                if 'entities' in entities_parent:
                    entities = entities_parent['entities']
                    if entities:
                        already_processed = all('nlp_analysis' in entity for entity in entities)
                        if already_processed:
                            nlp_data = entities[0]['nlp_analysis']
                    else:
                        already_processed = 'nlp_analysis' in content
                        if already_processed:
                            nlp_data = content['nlp_analysis']
                else:
                    already_processed = 'nlp_analysis' in content
                    if already_processed:
                        nlp_data = content['nlp_analysis']

                if already_processed and nlp_data:
                    print(f"\nUsing existing NLP analysis for node {node_id}")
                    # Use stored nlp_analysis for visualization
                    all_nlp_results.append(nlp_data)
                    # Extract data for visualizations
                    if 'sentiment' in enabled_analyses and 'sentiment' in nlp_data and nlp_data['sentiment']:
                        sentiment_scores.append({
                            'node_id': node_id,
                            'sentiment': nlp_data['sentiment']['label'],
                            'score': nlp_data['sentiment']['score']
                        })
                    # Add emotion data extraction
                    if 'emotion' in enabled_analyses and 'emotions' in nlp_data:
                        emotion_distributions.append({
                            'node_id': node_id,
                            'emotions': nlp_data['emotions']
                        })
                    if 'topic' in enabled_analyses and 'topics' in nlp_data:
                        topic_distributions.append({
                            'node_id': node_id,
                            'topics': nlp_data['topics']
                        })
                    if 'ner' in enabled_analyses and 'named_entities' in nlp_data:
                        for entity in nlp_data['named_entities']:
                            entity_counts[entity['label']] += 1
                    if 'top_bigrams' in nlp_data:
                        for bg in nlp_data['top_bigrams']:
                            bigram_tuple = tuple(bg['bigram'].split())
                            bigram_freq[bigram_tuple] += bg['count']
                    if 'top_trigrams' in nlp_data:
                        for tg in nlp_data['top_trigrams']:
                            trigram_tuple = tuple(tg['trigram'].split())
                            trigram_freq[trigram_tuple] += tg['count']
                    continue

                print(f"\nProcessing NLP for node: {node_id}")
                processed_nodes += 1
                try:
                    # [Rest of the processing loop remains unchanged]
                    # Create character-based chunks with overlap
                    chunks = []
                    start = 0
                    while start < len(full_text):
                        end = min(start + char_chunk_size, len(full_text))
                        if end < len(full_text):
                            split_point = full_text.rfind('.', start, end)
                            if split_point == -1:
                                split_point = full_text.rfind(' ', start, end)
                            if split_point != -1:
                                end = split_point + 1
                        chunk_text = full_text[start:end]
                        if chunk_text:
                            chunks.append(chunk_text)
                        start = end - char_chunk_overlap if end - char_chunk_overlap > start else end
                    if not chunks:
                        chunks = [full_text]
                    if max_chunks is not None:
                        chunks = chunks[:max_chunks]
                    # Initialize aggregators
                    aggregated_sentiment = {'positive': 0, 'negative': 0}
                    aggregated_emotions = defaultdict(float)  # Add emotion aggregator
                    aggregated_topics = defaultdict(float)
                    aggregated_entities = []
                    num_chunks = len(chunks)
                    language = 'en'
                    lang_confidence = 1.0
                    translated = False
                    for idx, text in enumerate(chunks):
                        print(f"Processing chunk {idx+1}/{num_chunks}: {text[:100]}...")
                        if 'language' in enabled_analyses and idx == 0:
                            start_time = time.time()
                            lang_result = language_pipeline(text, truncation=True, max_length=512)
                            language = lang_result[0]['label']
                            lang_confidence = lang_result[0]['score']
                            print(f"Time for language detection: {time.time() - start_time:.4f} seconds")
                        if 'translate' in enabled_analyses and language != 'en':
                            start_time = time.time()
                            trans_result = translate_pipeline(text, src_lang=language, tgt_lang='eng_Latn', truncation=True, max_length=512)
                            text = trans_result[0]['translation_text']
                            translated = True
                            print(f"Time for translation: {time.time() - start_time:.4f} seconds")
                        if 'sentiment' in enabled_analyses:
                            start_time = time.time()
                            sent_result = sentiment_pipeline(text, truncation=True, max_length=512)
                            label = sent_result[0]['label'].lower()
                            score = sent_result[0]['score']
                            aggregated_sentiment[label] += score
                            aggregated_sentiment['negative' if label == 'positive' else 'positive'] += (1 - score)
                            print(f"Time for sentiment analysis: {time.time() - start_time:.4f} seconds")
                        # Add emotion recognition processing
                        if 'emotion' in enabled_analyses:
                            start_time = time.time()
                            emotion_result = emotion_pipeline(text, truncation=True, max_length=512)
                            for e in emotion_result[0]:
                                aggregated_emotions[e['label']] += e['score']
                            print(f"Time for emotion recognition: {time.time() - start_time:.4f} seconds")
                        if 'topic' in enabled_analyses:
                            start_time = time.time()
                            topic_labels = [
                                "technology", "politics", "sports", "entertainment", "business",
                                "health", "education", "travel", "food", "lifestyle", "science",
                                "environment", "fashion", "art", "music", "gaming"
                            ]
                            topic_result = zero_shot_pipeline(text, topic_labels, truncation=True, max_length=1024)
                            for label, score in zip(topic_result['labels'][:5], topic_result['scores'][:5]):
                                aggregated_topics[label] += score
                            print(f"Time for topic classification: {time.time() - start_time:.4f} seconds")
                        if 'ner' in enabled_analyses:
                            start_time = time.time()
                            ner_results = ner_pipeline(text)
                            for e in ner_results:
                                entity_dict = {"text": e['word'], "label": e['entity_group'], "score": e['score']}
                                if entity_dict not in aggregated_entities:
                                    aggregated_entities.append(entity_dict)
                                entity_counts[e['entity_group']] += 1
                            print(f"Time for NER: {time.time() - start_time:.4f} seconds")
                    # Aggregate results
                    if 'sentiment' in enabled_analyses:
                        avg_pos = aggregated_sentiment['positive'] / num_chunks
                        avg_neg = aggregated_sentiment['negative'] / num_chunks
                        sentiment = 'POSITIVE' if avg_pos > avg_neg else 'NEGATIVE'
                        sent_score = max(avg_pos, avg_neg)
                        sentiment_scores.append({'node_id': node_id, 'sentiment': sentiment, 'score': sent_score})
                    else:
                        sentiment = None
                        sent_score = 0
                    # Add emotion aggregation
                    if 'emotion' in enabled_analyses:
                        emotions = {k: v / num_chunks for k, v in aggregated_emotions.items()}
                        emotion_distributions.append({'node_id': node_id, 'emotions': emotions})
                    else:
                        emotions = {}
                    if 'topic' in enabled_analyses:
                        topics = [{"label": k, "score": v / num_chunks} for k, v in sorted(aggregated_topics.items(), key=lambda x: x[1], reverse=True)[:5]]
                        topic_distributions.append({'node_id': node_id, 'topics': topics})
                    else:
                        topics = []
                    if 'ner' in enabled_analyses:
                        named_entities = aggregated_entities
                    else:
                        named_entities = []
                    top_bigrams = Counter(bigrams).most_common(10)
                    top_trigrams = Counter(trigrams).most_common(10)
                    # Compile NLP data
                    nlp_data = {}
                    if 'language' in enabled_analyses:
                        nlp_data["language"] = {"code": language, "confidence": lang_confidence}
                    if 'translate' in enabled_analyses:
                        nlp_data["translated"] = translated
                    if 'sentiment' in enabled_analyses:
                        nlp_data["sentiment"] = {"label": sentiment, "score": sent_score}
                    # Add emotion data to nlp_data
                    if 'emotion' in enabled_analyses:
                        nlp_data["emotions"] = emotions
                    if 'topic' in enabled_analyses:
                        nlp_data["topics"] = topics
                    if 'ner' in enabled_analyses:
                        nlp_data["named_entities"] = named_entities
                    nlp_data["top_bigrams"] = [{"bigram": ' '.join(bg[0]), "count": bg[1]} for bg in top_bigrams]
                    nlp_data["top_trigrams"] = [{"trigram": ' '.join(tg[0]), "count": tg[1]} for tg in top_trigrams]
                    nlp_data["text_length"] = len(full_text)
                    nlp_data["word_count"] = len(words)
                    nlp_data = convert_to_serializable(nlp_data)
                    all_nlp_results.append(nlp_data)
                    # Update DB
                    found = False
                    entities_parent = content
                    if 'profileData' in content:
                        for key in content['profileData']:
                            if 'entities' in content['profileData'][key]:
                                entities_parent = content['profileData'][key]
                                break
                    if 'entities' in entities_parent:
                        entities = entities_parent['entities']
                        for entity in entities:
                            entity['nlp_analysis'] = nlp_data
                            found = True
                    if not found:
                        content['nlp_analysis'] = nlp_data
                        found = True
                    if found:
                        update_query = f'UPDATE "{project}_nodes" SET content = %s WHERE id = %s;'
                        cursor.execute(update_query, (json.dumps(content), node_id))
                        connection.commit()
                        print(f"Updated NLP analysis for node {node_id}")
                    else:
                        print(f"No place to add NLP analysis for node {node_id}")
                except Exception as e:
                    print(f"Error processing NLP for node {node_id}: {e}")

            # Global aggregates
            # No need to redefine bigram_freq and trigram_freq here
            word_freq = Counter(all_words)

            # Print top words, bigrams, trigrams
            print("\n=== Global Text Statistics ===")
            print("Top 20 most common words:")
            for word, count in word_freq.most_common(20):
                print(f"{word}: {count}")
            print("\nTop 20 most common bigrams:")
            for bg, count in bigram_freq.most_common(20):
                print(f"{' '.join(bg)}: {count}")
            print("\nTop 20 most common trigrams:")
            for tg, count in trigram_freq.most_common(20):
                print(f"{' '.join(tg)}: {count}")

            # Visualization section (enhanced with emotion analysis)
            print("\n=== Creating NLP Visualizations ===")
            graphs_dir = os.path.join(project_dir, 'graphs')
            os.makedirs(graphs_dir, exist_ok=True)
            visualization_paths = []
            # Sentiment Distribution
            if 'sentiment' in enabled_analyses and sentiment_scores:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                sentiment_counts = Counter([s['sentiment'] for s in sentiment_scores])
                ax1.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', startangle=90)
                ax1.set_title('Sentiment Distribution Across Nodes')
                scores = [s['score'] for s in sentiment_scores]
                ax2.hist(scores, bins=20, edgecolor='black', alpha=0.7)
                ax2.set_xlabel('Sentiment Confidence Score')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Distribution of Sentiment Confidence Scores')
                sentiment_path = os.path.join(graphs_dir, 'sentiment_analysis.png')
                plt.tight_layout()
                plt.savefig(sentiment_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths.append(sentiment_path)

            # Add Emotion Analysis Heatmap from second script
            if 'emotion' in enabled_analyses and emotion_distributions:
                fig, ax = plt.subplots(figsize=(12, 8))
                emotion_names = set()
                for ed in emotion_distributions:
                    emotion_names.update(ed['emotions'].keys())
                emotion_names = sorted(list(emotion_names))
                emotion_matrix = []
                node_ids = []
                for ed in emotion_distributions[:20]:  # Limit to first 20 nodes for readability
                    node_ids.append(f"Node {ed['node_id']}")
                    row = [ed['emotions'].get(emotion, 0) for emotion in emotion_names]
                    emotion_matrix.append(row)
                if emotion_matrix:
                    im = ax.imshow(emotion_matrix, cmap='YlOrRd', aspect='auto')
                    ax.set_xticks(range(len(emotion_names)))
                    ax.set_xticklabels(emotion_names, rotation=45, ha='right')
                    ax.set_yticks(range(len(node_ids)))
                    ax.set_yticklabels(node_ids)
                    ax.set_xlabel('Emotions')
                    ax.set_ylabel('Nodes')
                    ax.set_title('Emotion Distribution Heatmap')
                    plt.colorbar(im, ax=ax, label='Emotion Score')
                emotion_path = os.path.join(graphs_dir, 'emotion_heatmap.png')
                plt.tight_layout()
                plt.savefig(emotion_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths.append(emotion_path)

            # Topic Distribution
            if 'topic' in enabled_analyses and topic_distributions:
                fig, ax = plt.subplots(figsize=(12, 8))
                topic_scores = defaultdict(list)
                for td in topic_distributions:
                    for topic in td['topics']:
                        topic_scores[topic['label']].append(topic['score'])
                topic_avg = {topic: np.mean(scores) for topic, scores in topic_scores.items()}
                sorted_topics = sorted(topic_avg.items(), key=lambda x: x[1], reverse=True)[:10]
                topics = [t[0] for t in sorted_topics]
                scores = [t[1] for t in sorted_topics]
                bars = ax.barh(topics, scores, color=plt.cm.viridis(np.linspace(0, 1, len(topics))))
                ax.set_xlabel('Average Score')
                ax.set_title('Top 10 Topics Across All Nodes')
                ax.set_xlim(0, 1)
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', ha='left', va='center')
                topic_path = os.path.join(graphs_dir, 'topic_distribution.png')
                plt.tight_layout()
                plt.savefig(topic_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths.append(topic_path)
            # Named Entity Distribution
            if 'ner' in enabled_analyses and entity_counts:
                fig, ax = plt.subplots(figsize=(10, 6))
                sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
                if sorted_entities:
                    entities = [e[0] for e in sorted_entities]
                    counts = [e[1] for e in sorted_entities]
                    ax.bar(entities, counts, color='skyblue', edgecolor='navy')
                    ax.set_xlabel('Entity Type')
                    ax.set_ylabel('Count')
                    ax.set_title('Named Entity Recognition Results')
                    ax.tick_params(axis='x', rotation=45)
                ner_path = os.path.join(graphs_dir, 'named_entities.png')
                plt.tight_layout()
                plt.savefig(ner_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths.append(ner_path)
            # Language Distribution
            if 'language' in enabled_analyses:
                fig, ax = plt.subplots(figsize=(10, 6))
                language_counts = Counter([r['language']['code'] for r in all_nlp_results if 'language' in r])
                if language_counts:
                    languages = list(language_counts.keys())
                    counts = list(language_counts.values())
                    ax.pie(counts, labels=languages, autopct='%1.1f%%', startangle=90)
                    ax.set_title('Language Distribution Across Nodes')
                language_path = os.path.join(graphs_dir, 'language_distribution.png')
                plt.tight_layout()
                plt.savefig(language_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths.append(language_path)
            # Word Cloud
            if word_freq:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
                plt.figure(figsize=(10, 6))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis("off")
                plt.title("Word Cloud of Frequent Words")
                wordcloud_path = os.path.join(graphs_dir, 'word_cloud.png')
                plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths.append(wordcloud_path)
            # Word Frequency Bar Chart
            fig, ax = plt.subplots(figsize=(12, 8))
            top_words = word_freq.most_common(20)
            if top_words:
                words = [w[0] for w in top_words]
                counts = [w[1] for w in top_words]
                ax.barh(words, counts, color='green')
                ax.set_xlabel('Frequency')
                ax.set_title('Top 20 Most Common Words')
                ax.invert_yaxis()
            word_freq_path = os.path.join(graphs_dir, 'word_frequency.png')
            plt.tight_layout()
            plt.savefig(word_freq_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths.append(word_freq_path)
            # Bigram Frequency Bar Chart
            fig, ax = plt.subplots(figsize=(12, 8))
            top_bigrams_global = bigram_freq.most_common(20)
            if top_bigrams_global:
                bigrams_str = [' '.join(b[0]) for b in top_bigrams_global]
                counts = [b[1] for b in top_bigrams_global]
                ax.barh(bigrams_str, counts, color='blue')
                ax.set_xlabel('Frequency')
                ax.set_title('Top 20 Most Common Bigrams')
                ax.invert_yaxis()
            bigram_path = os.path.join(graphs_dir, 'bigram_frequency.png')
            plt.tight_layout()
            plt.savefig(bigram_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths.append(bigram_path)
            # Trigram Frequency Bar Chart
            fig, ax = plt.subplots(figsize=(12, 8))
            top_trigrams_global = trigram_freq.most_common(20)
            if top_trigrams_global:
                trigrams_str = [' '.join(t[0]) for t in top_trigrams_global]
                counts = [t[1] for t in top_trigrams_global]
                ax.barh(trigrams_str, counts, color='purple')
                ax.set_xlabel('Frequency')
                ax.set_title('Top 20 Most Common Trigrams')
                ax.invert_yaxis()
            trigram_path = os.path.join(graphs_dir, 'trigram_frequency.png')
            plt.tight_layout()
            plt.savefig(trigram_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths.append(trigram_path)
            # Send visualizations to endpoint
            print("\n=== Sending NLP visualizations to endpoint ===")
            images = []
            for viz_path in visualization_paths:
                if os.path.exists(viz_path):
                    with open(viz_path, 'rb') as f:
                        img_data = f.read()
                        base64_str = base64.b64encode(img_data).decode('utf-8')
                        images.append(base64_str)
                        print(f"Prepared visualization: {os.path.basename(viz_path)}")
            payload = {"images": images}
            try:
                response = requests.post('http://127.0.0.1:4000/python-images', json=payload)
                if response.status_code == 200:
                    print(f"Successfully sent {len(images)} visualizations")
                else:
                    print(f"Failed to send visualizations: {response.status_code}, {response.text}")
            except Exception as e:
                print(f"Error sending visualizations: {e}")
            print(f"\nGenerated {len(visualization_paths)} NLP visualizations")
            print("\n=== NLP Analysis Complete ===")
            print(f"Processed {processed_nodes} nodes")
            print(f"Generated {len(visualization_paths)} visualizations")
    if 'object_detection' in analyses_list:
        visualization_paths = []
        all_detections = [] # Collect all detections for summary statistics
        detection_model = load_detection_model()
        image_count = 0
        max_images_for_bbox = 5 # Limit the number of images with bounding box visualizations
        for filename in os.listdir(image_folder):
            image_path = os.path.join(image_folder, filename)
            if not os.path.isfile(image_path):
                continue
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                print(f"Skipping non-image file: {filename}")
                continue
            print(f"\nProcessing image for object detection: {filename}")
            parts = filename[:-4].split('_')
            if len(parts) < 4 or parts[0] != 'node' or parts[2] != 'entity':
                print(f"Skipping invalid filename for detection: {filename}")
                continue
            node_id = parts[1]
            entity_idx = int(parts[3])
            is_frame = False
            frame_idx = None
            if len(parts) > 4 and parts[4] == 'frame':
                if len(parts) > 5:
                    is_frame = True
                    frame_idx = int(parts[5])
                else:
                    continue
            try:
                select_query = f'SELECT content FROM "{project}_nodes" WHERE id = %s;'
                cursor.execute(select_query, (node_id,))
                row = cursor.fetchone()
                if not row:
                    print(f"No node found for id {node_id}")
                    continue
                content = row[0]
                found = False
                detections = None
                update_needed = False
                if 'image' in content and entity_idx == 1 and not is_frame:
                    if 'detections' in content:
                        detections = content['detections']
                        print(f"Using existing detections for node {node_id}, entity {entity_idx}")
                    else:
                        detections = detect_objects(image_path, detection_model)
                        content['detections'] = detections
                        update_needed = True
                    found = True
                else:
                    entities_parent = content
                    if 'profileData' in content:
                        for key in content['profileData']:
                            if 'entities' in content['profileData'][key]:
                                entities_parent = content['profileData'][key]
                                break
                    if 'entities' in entities_parent:
                        entities = entities_parent['entities']
                        for entity in entities:
                            if entity.get('index') == entity_idx:
                                if is_frame:
                                    if 'frame_detections' not in entity:
                                        entity['frame_detections'] = {}
                                    if frame_idx in entity['frame_detections']:
                                        detections = entity['frame_detections'][frame_idx]
                                        print(f"Using existing detections for node {node_id}, entity {entity_idx}, frame {frame_idx}")
                                    else:
                                        detections = detect_objects(image_path, detection_model)
                                        entity['frame_detections'][frame_idx] = detections
                                        update_needed = True
                                else:
                                    if 'detections' in entity:
                                        detections = entity['detections']
                                        print(f"Using existing detections for node {node_id}, entity {entity_idx}")
                                    else:
                                        detections = detect_objects(image_path, detection_model)
                                        entity['detections'] = detections
                                        update_needed = True
                                found = True
                                break
                if not found:
                    print(f"No place to add detections for node {node_id}, entity {entity_idx}")
                    continue
                if update_needed:
                    update_query = f'UPDATE "{project}_nodes" SET content = %s WHERE id = %s;'
                    cursor.execute(update_query, (json.dumps(content), node_id))
                    connection.commit()
                    print(f"Updated detections for node {node_id}, entity {entity_idx}" + (f", frame {frame_idx}" if is_frame else ""))
                if detections:
                    all_detections.extend(detections)
                # Generate bounding box visualization for the first few images
                if detections and image_count < max_images_for_bbox:
                    bbox_path = os.path.join(graphs_dir, f'bbox_{filename}')
                    if os.path.exists(bbox_path):
                        print(f"Using existing visualization: {bbox_path}")
                    else:
                        img = cv2.imread(image_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        for det in detections:
                            bbox = det['bbox'][0]
                            x1, y1, x2, y2 = map(int, bbox)
                            label = det['label']
                            confidence = det['confidence']
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.imshow(img)
                        ax.axis('off')
                        ax.set_title(f'Objects in {filename}')
                        plt.savefig(bbox_path, bbox_inches='tight')
                        plt.close()
                    visualization_paths.append(bbox_path)
                    image_count += 1
            except Exception as e:
                print(f"Error updating detections for {filename}: {e}")
        # Generate summary visualizations
        if all_detections:
            graphs_dir = os.path.join(project_dir, 'graphs')
            os.makedirs(graphs_dir, exist_ok=True)
            # Histogram of object classes
            object_counts = Counter([det['label'] for det in all_detections])
            if object_counts:
                obj_hist_path = os.path.join(graphs_dir, 'object_class_hist.png')
                if not os.path.exists(obj_hist_path):
                    fig, ax = plt.subplots(figsize=(12, 6))
                    labels, counts = zip(*object_counts.items())
                    ax.bar(labels, counts, color='skyblue')
                    ax.set_xlabel('Object Class')
                    ax.set_ylabel('Count')
                    ax.set_title('Distribution of Detected Object Classes')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(obj_hist_path)
                    plt.close()
                visualization_paths.append(obj_hist_path)
            # Confidence score distribution
            confidences = [det['confidence'] for det in all_detections]
            conf_hist_path = os.path.join(graphs_dir, 'object_confidence_hist.png')
            if not os.path.exists(conf_hist_path):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(confidences, bins=30, color='blue', alpha=0.7)
                ax.set_xlabel('Confidence Score')
                ax.set_ylabel('Frequency')
                ax.set_title('Distribution of Object Detection Confidence Scores')
                plt.savefig(conf_hist_path)
                plt.close()
            visualization_paths.append(conf_hist_path)
            # Pie chart of object class proportions
            if object_counts:
                obj_pie_path = os.path.join(graphs_dir, 'object_class_pie.png')
                if not os.path.exists(obj_pie_path):
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
                    ax.set_title('Proportion of Detected Object Classes')
                    plt.savefig(obj_pie_path)
                    plt.close()
                visualization_paths.append(obj_pie_path)
        # Add visualizations to images list
        for viz_path in visualization_paths:
            if os.path.exists(viz_path):
                with open(viz_path, 'rb') as f:
                    img_data = f.read()
                    base64_str = base64.b64encode(img_data).decode('utf-8')
                    images.append(base64_str) # Add to images list
                    print(f"Prepared visualization: {os.path.basename(viz_path)}")
        # Send the list of base64 images
        payload = {"images": images}
        try:
            response = requests.post('http://127.0.0.1:4000/python-images', json=payload)
            if response.status_code == 200:
                print(f"Successfully sent {len(images)} visualizations")
            else:
                print(f"Failed to send visualizations: {response.status_code}, {response.text}")
        except Exception as e:
            print(f"Error sending visualizations: {e}")
        print(f"\nGenerated {len(visualization_paths)} object detection visualizations")
        print("\n=== NLP Analysis Complete ===")
        print(f"Processed {len(all_nlp_results)} nodes")
        print(f"Generated {len(visualization_paths)} visualizations")
    if 'geo_analysis' in analyses_list:
        print("\nProcessing location entities for geocoding...")
        model = GeoCLIP()
        for filename in os.listdir(image_folder):
            image_path = os.path.join(image_folder, filename)
            if not os.path.isfile(image_path):
                continue
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                print(f"Skipping non-image file: {filename}")
                continue
            print(f"\nProcessing image: {filename}")
            process_image(image_path, endpoint_url, output_file, model, additional_info, project)

        # Collect all geolocation data from the database for plotting
        print("\nCollecting geolocation data for visualizations...")
        query = f"""SELECT id, content FROM "{project}_nodes";"""
        cursor.execute(query)
        rows = cursor.fetchall()
        geo_data = []
        for node_id, content in rows:
            if 'geolocation' in content:
                geo = content['geolocation'].copy()
                geo['node_id'] = node_id
                geo_data.append(geo)
            entities_parent = content
            if 'profileData' in content:
                for key in content['profileData']:
                    if 'entities' in content['profileData'][key]:
                        entities_parent = content['profileData'][key]
                        break
            if 'entities' in entities_parent:
                for entity in entities_parent['entities']:
                    if 'geolocation' in entity:
                        geo = entity['geolocation'].copy()
                        geo['node_id'] = node_id
                        geo['entity_idx'] = entity.get('index')
                        geo_data.append(geo)
                    if 'frame_geolocation' in entity:
                        for frame_idx, fgeo in entity['frame_geolocation'].items():
                            geo = fgeo.copy()
                            geo['node_id'] = node_id
                            geo['entity_idx'] = entity.get('index')
                            geo['frame_idx'] = frame_idx
                            geo_data.append(geo)

        if geo_data:
            graphs_dir = os.path.join(project_dir, 'graphs')
            os.makedirs(graphs_dir, exist_ok=True)
            visualization_paths = []

            # 3. Histogram of geolocation probabilities
            probs = [d['probability'] for d in geo_data if 'probability' in d and d['probability'] is not None]
            if probs:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(probs, bins=20, color='blue', edgecolor='black')
                ax.set_title('Distribution of Geolocation Probabilities')
                ax.set_xlabel('Probability')
                ax.set_ylabel('Count')
                prob_path = os.path.join(graphs_dir, 'geo_probabilities.png')
                plt.tight_layout()
                plt.savefig(prob_path, dpi=300)
                plt.close()
                visualization_paths.append(prob_path)
                print(f"Generated probabilities histogram: {prob_path}")

            # Send visualizations as base64 to endpoint
            images = []
            for viz_path in visualization_paths:
                if os.path.exists(viz_path):
                    with open(viz_path, 'rb') as f:
                        img_data = f.read()
                        base64_str = base64.b64encode(img_data).decode('utf-8')
                        images.append(base64_str)
                    print(f"Prepared geo visualization: {os.path.basename(viz_path)}")

            payload = {"images": images}
            try:
                response = requests.post('http://127.0.0.1:4000/python-images', json=payload)
                if response.status_code == 200:
                    print(f"Successfully sent {len(images)} geo visualizations")
                else:
                    print(f"Failed to send geo visualizations: {response.status_code}, {response.text}")
            except Exception as e:
                print(f"Error sending geo visualizations: {e}")

        else:
            print("No geolocation data found for visualizations.")
    if 'network_analysis' in analyses_list:
        print(f"\nPerforming network analysis for project: {project}")
        try:
            # Query nodes and edges
            cursor.execute(f'SELECT id, s_id, content, page_text FROM "{project}_nodes";')
            nodes = cursor.fetchall()
            cursor.execute(f'SELECT from_id, to_id FROM "{project}_edges";')
            edges_list = cursor.fetchall()
            # Create graph
            G = nx.DiGraph()
            for node_id, s_id, content, page_text in nodes:
                G.add_node(node_id, s_id=s_id, content=content, page_text=page_text)
            for from_id, to_id in edges_list:
                G.add_edge(from_id, to_id)
            # Compute centralities
            btwn = nx.betweenness_centrality(G)
            close = nx.closeness_centrality(G)
            in_deg = nx.in_degree_centrality(G)
            out_deg = nx.out_degree_centrality(G)
            clust = nx.clustering(G)
            try:
                A = nx.to_numpy_array(G)  # Use numpy array for efficiency
                eigenvalues = scipy.linalg.eigvals(A)
                lambda_max = np.max(np.abs(eigenvalues.real))
                alpha = 0.85 / (lambda_max + 1e-6) if lambda_max > 0 else 0.01
                katz = nx.katz_centrality(G, alpha=alpha, max_iter=10000, tol=1e-03)
            except (nx.NetworkXException, ValueError) as e:
                print(f"Failed to compute Katz (eigenvalue issue: {e}). Using small alpha fallback.")
                try:
                    katz = nx.katz_centrality(G, alpha=0.01, max_iter=10000, tol=1e-03)
                except nx.PowerIterationFailedConvergence as e:
                    print(f"Katz centrality failed: {e}. Setting all to 0.")
                    katz = {node: 0.0 for node in G.nodes}

            avg_cluster = nx.average_clustering(G)
            louvain = nx.community.louvain_communities(G)
            print(len(louvain))
            if not nx.is_strongly_connected(G):
                print("Graph is not strongly connected. Computing eigenvector centrality per component.")
                eigen = {}
                for component in nx.strongly_connected_components(G):
                    subgraph = G.subgraph(component)
                    if len(subgraph) > 1:  # Skip trivial components
                        try:
                            comp_eigen = nx.eigenvector_centrality(subgraph, max_iter=10000, tol=1e-03)
                        except nx.NetworkXException as e:
                            print(f"Power iteration failed for component {component}: {e}. Setting to 0.")
                            comp_eigen = {node: 0.0 for node in component}
                        eigen.update(comp_eigen)
                    else:
                        eigen.update({node: 0.0 for node in component})
            else:
                try:
                    eigen = nx.eigenvector_centrality(G, max_iter=10000, tol=1e-03)
                except nx.NetworkXException as e:
                    print(f"Power iteration failed for full graph: {e}. Setting all to 0.")
                    eigen = {node: 0.0 for node in G.nodes}
            # Update DB
            for node_id in G.nodes:
                cent = {
                    'betweenness': btwn.get(node_id, 0),
                    'closeness': close.get(node_id, 0),
                    'in_degree': in_deg.get(node_id, 0),
                    'out_degree': out_deg.get(node_id, 0),
                    'eigenvector': eigen.get(node_id, 0),
                    'cluster': clust.get(node_id, 0),
                    'katz': katz.get(node_id, 0),
                }
                try:
                    cursor.execute(f'SELECT content FROM "{project}_nodes" WHERE id = %s;', (node_id,))
                    row = cursor.fetchone()
                    if row:
                        content = row[0]
                        content['centralities'] = cent
                        cursor.execute(f'UPDATE "{project}_nodes" SET content = %s WHERE id = %s;', (json.dumps(content), node_id))
                        connection.commit()
                except Exception as e:
                    print(f"Error updating centralities for node {node_id}: {e}")
            print(len(G.nodes))
            print(len(G.edges))
            # Generate graphs/images (adapted from Jupyter)
            # Histogram of centralities
            centralities_dict = {}
            for node_id in G.nodes:
                cent = {
                    'betweenness': btwn.get(node_id, 0),
                    'closeness': close.get(node_id, 0),
                    'in_degree': in_deg.get(node_id, 0),
                    'out_degree': out_deg.get(node_id, 0),
                    'eigenvector': eigen.get(node_id, 0),
                    'cluster': clust.get(node_id, 0),
                    'katz': katz.get(node_id, 0),
                }
                centralities_dict[node_id] = cent

            images = []
            if centralities_dict:
                df = pd.DataFrame.from_dict(centralities_dict, orient='index')
                columns_labels = {
                    "betweenness": "Betweenness",
                    "out_degree": "Out-degree",
                    "closeness": "Closeness",
                    "in_degree": "In-degree",
                    "eigenvector": "Eigenvector",
                    "cluster": "Cluster",
                    "katz": "Katz",
                }
                fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))
                axes = axes.flatten()
                for ax, cl in zip(axes, columns_labels.items()):
                    col, label = cl
                    X = df[col].values
                    hist, bins = np.histogram(X, bins=10)
                    hist = hist * 100 / np.sum(hist) if np.sum(hist) > 0 else hist
                    ax.bar(bins[:-1], hist, width=np.diff(bins), edgecolor="black", align="edge", alpha=0.7, color=plt.get_cmap("Paired")(9))
                    ax.set_xlabel(f"Normalized {label}")
                    ax.set_yscale("log")
                fig.supylabel("Number of cells [%]")
                graphs_dir = os.path.join(project_dir, 'graphs')
                os.makedirs(graphs_dir, exist_ok=True)
                hist_path = os.path.join(graphs_dir, 'centralities_hist.png')
                fig.savefig(hist_path)
                plt.close(fig)  # Free memory
                with open(hist_path, 'rb') as f:
                    img_data = f.read()
                    base64_str = base64.b64encode(img_data).decode('utf-8')
                    images.append(base64_str)
            else:
                print("No centralities to plot.")

            # Extend with geo data and additional plots
            geo_data = []
            for node_id, s_id, content, page_text in nodes:
                # Debug: Check for image-related data in content
                if any(key in content for key in ['image', 'img', 'thumbnail', 'profile_image']):
                    print(f"Node {node_id}: Content keys with potential image data: {list(content.keys())}")
                lat = lon = None
                if 'geolocation' in content:
                    geo = content['geolocation']
                    lat = geo.get('lat')
                    lon = geo.get('lon')
                entities_parent = content
                if 'profileData' in content:
                    for key in content['profileData']:
                        if 'entities' in content['profileData'][key]:
                            entities_parent = content['profileData'][key]
                            break
                if 'entities' in entities_parent:
                    entities = entities_parent['entities']
                    for entity in entities:
                        if 'geolocation' in entity:
                            lat = entity['geolocation'].get('lat')
                            lon = entity['geolocation'].get('lon')
                            break
                if lat is not None and lon is not None:
                    lang = None
                    flg = 0
                    geocoded_location = None
                    if 'nlp_analysis' in content:
                        lang = content['nlp_analysis']['language']['code']
                    if 'profileData' in content:
                        for key in content['profileData']:
                            profile = content['profileData'][key]
                            if 'followers_count' in profile:
                                flg = profile['followers_count']
                                break
                    if 'location' in content:
                        geocoded_location = content['location']
                    geo_data.append({
                        '_id': node_id,
                        'lat': lat,
                        'lon': lon,
                        'lang': lang,
                        'geocoded_location': geocoded_location,
                        'flg': flg
                    })

            if geo_data:
                try:
                    df = pd.DataFrame(geo_data)
                    df = df.set_index('_id')
                    for col, cent in zip(['betweenness', 'closeness', 'in_degree', 'out_degree', 'eigenvector', 'cluster', 'katz'], [btwn, close, in_deg, out_deg, eigen, clust, katz]):
                        df[col] = df.index.map(cent)
                    geometry = [shp_geom.Point(x, y) for x, y in df[["lon", "lat"]].values]
                    df["geometry"] = geometry
                    gdf = gpd.GeoDataFrame(df, geometry="geometry")
                    if not gdf.empty:
                        df_coordinates = gdf[["lon", "lat"]].rename(columns={"lon": "x", "lat": "y"})
                        df_edges_graph = pd.DataFrame(list(G.edges), columns=["source", "target"])
                        df_bundled = hammer_bundle(df_coordinates, df_edges_graph)
                        c_background = "#1D2224"
                        c_countries = "#393A3A"
                        c_borders = "white"
                        try:
                            countries = gpd.read_file("./ne_10m_admin_0_countries/")
                            fig, ax = plt.subplots(figsize=(16, 9))
                            fig.set_facecolor(c_background)
                            countries.plot(ax=ax, color=c_countries, edgecolor=c_borders, linewidth=0.1)
                            ax.plot(df_coordinates.x, df_coordinates.y, marker="o", markersize=1, color=plt.get_cmap("Paired")(6), linestyle="None", alpha=0.45)
                            ax.plot(df_bundled.x, df_bundled.y, color=plt.get_cmap("Paired")(9), alpha=0.85, linewidth=0.1)
                            ax.axis("off")
                            handles = [
                                mpl_patches.Patch(facecolor=plt.get_cmap("Paired")(9), label="Followership"),
                                mpl_patches.Patch(facecolor=plt.get_cmap("Paired")(6), label="Gettr Users in Network"),
                            ]
                            ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.2, 0.2))
                            path = os.path.join(graphs_dir, 'nodesundedges.png')
                            fig.savefig(path, dpi=900)
                            plt.close(fig)  # Free memory
                            with open(path, 'rb') as f:
                                img_data = f.read()
                                base64_str = base64.b64encode(img_data).decode('utf-8')
                                images.append(base64_str)
                        except Exception as e:
                            print(f"Error in geoplotting: {e}. Skipping map plot.")
                    else:
                        print("No valid geolocation data after dropping NaN geocoded_location.")
                except Exception as e:
                    print(f"Error processing geo data: {e}. Skipping geoplotting.")
            else:
                print("No valid geolocation data found; skipping map plot.")

            # Send all images
            if images:
                response = requests.post('http://127.0.0.1:4000/python-images', json={'images': images})
                if response.status_code == 200:
                    print(f"Python graph images sent successfully ({len(images)} images)")
                else:
                    print(f"Error sending Python graph images: {response.status_code}")
            else:
                print("No images generated for sending.")
            print("Network analysis completed.")
        except Exception as e:
            print(f"Error during network analysis: {e}")
    if 'face_recognition' in analyses_list:
            processed_files = set()
            # Track if ANY file was newly processed (for deciding whether to regenerate plots)
            any_new_processing = False

            for filename in os.listdir(image_folder):
                image_path = os.path.join(image_folder, filename)

                # Skip non-image files
                if not os.path.isfile(image_path) or not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    print(f"Skipping non-image file: {filename}")
                    continue

                # Parse filename (e.g., node_<node_id>_entity_<entity_idx>[_frame_<frame_idx>].jpg)
                parts = filename.rsplit('.', 1)[0].split('_')
                if len(parts) < 4 or parts[0] != 'node' or parts[2] != 'entity':
                    print(f"Skipping invalid filename for face recognition: {filename}")
                    continue

                node_id = parts[1]
                try:
                    entity_idx = int(parts[3])
                except ValueError:
                    print(f"Invalid entity index in filename: {filename}")
                    continue

                is_frame = len(parts) > 4 and parts[4] == 'frame'
                frame_idx = str(parts[5]) if is_frame and len(parts) > 5 else None

                # Log parsing details for debugging
                print(f"Parsed filename: {filename} -> node_id={node_id}, entity_idx={entity_idx}, is_frame={is_frame}, frame_idx={frame_idx}")

                # Check database for existing analyses
                try:
                    select_query = f'SELECT content FROM "{project}_nodes" WHERE id = %s;'
                    cursor.execute(select_query, (node_id,))
                    row = cursor.fetchone()
                    if not row:
                        print(f"No node found for id {node_id}")
                        continue

                    content = row[0]
                    skip_processing = False
                    found_place = False

                    # Check file modification time
                    file_mtime = os.path.getmtime(image_path)

                    # Check for existing face analyses
                    if 'image' in content and entity_idx == 1 and not is_frame:
                        found_place = True
                        if 'face_analyses' in content and content.get('last_processed', 0) >= file_mtime:
                            print(f"Face analyses exist and file unchanged for node {node_id}, entity {entity_idx}, skipping processing")
                            skip_processing = True
                            # Mark as processed for tracking (but will be included in plots)
                            processed_files.add(filename)
                    else:
                        entities_parent = content
                        if 'profileData' in content:
                            for key in content['profileData']:
                                if 'entities' in content['profileData'][key]:
                                    entities_parent = content['profileData'][key]
                                    break
                        if 'entities' in entities_parent:
                            for entity in entities_parent['entities']:
                                if entity.get('index') == entity_idx:
                                    found_place = True
                                    if is_frame:
                                        if ('frame_face_analyses' in entity and
                                            frame_idx in entity['frame_face_analyses'] and
                                            entity.get('last_processed', 0) >= file_mtime):
                                            print(f"Frame face analyses exist and file unchanged for node {node_id}, entity {entity_idx}, frame {frame_idx}, skipping processing")
                                            skip_processing = True
                                            # Mark as processed for tracking
                                            processed_files.add(filename)
                                    else:
                                        if 'face_analyses' in entity and entity.get('last_processed', 0) >= file_mtime:
                                            print(f"Face analyses exist and file unchanged for node {node_id}, entity {entity_idx}, skipping processing")
                                            skip_processing = True
                                            # Mark as processed for tracking
                                            processed_files.add(filename)
                                    break

                    if not found_place:
                        print(f"No place found to add face analyses for {filename}, skipping")
                        continue

                    if skip_processing:
                        # Already marked as processed above, just continue
                        continue

                    # If we get here, we need to process this image
                    print(f"\nProcessing image for face recognition: {filename}")
                    face_analyses = []
                    try:
                        # Perform face analysis
                        face_analyses = DeepFace.analyze(image_path, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)
                        if face_analyses:
                            print(f"Detected {len(face_analyses)} faces in {filename}:")
                            for i, face in enumerate(face_analyses):
                                print(f"Face {i+1}:")
                                print(f" Region: {face.get('region', 'N/A')}")
                                print(f" Age: {face.get('age', 'N/A')}")
                                print(f" Dominant Gender: {face.get('dominant_gender', 'N/A')}")
                                print(f" Dominant Emotion: {face.get('dominant_emotion', 'N/A')}")
                                print(f" Dominant Race: {face.get('dominant_race', 'N/A')}")
                        else:
                            print(f"No faces detected in {filename}")

                        # Get embeddings
                        try:
                            embeddings_objs = DeepFace.represent(image_path, enforce_detection=False)
                            if len(face_analyses) == len(embeddings_objs):
                                for i in range(len(face_analyses)):
                                    face_analyses[i]['embedding'] = embeddings_objs[i]['embedding']
                                    if uploaded_face_embedding is not None:
                                        face_emb = np.array(face_analyses[i]['embedding'])
                                        norm_face = np.linalg.norm(face_emb)
                                        norm_uploaded = np.linalg.norm(uploaded_face_embedding)
                                        dist = 1.0 if norm_face == 0 or norm_uploaded == 0 else ssd_cosine(face_emb / norm_face, uploaded_face_embedding / norm_uploaded)
                                        if dist < 0.68:
                                            face_analyses[i]['matches_uploaded_face'] = True
                                            face_analyses[i]['match_distance'] = float(dist)
                                            print(f"Face {i+1} matches uploaded face (distance: {dist:.4f})")
                                        else:
                                            face_analyses[i]['matches_uploaded_face'] = False
                        except Exception as embed_e:
                            print(f"Error getting embeddings for {filename}: {embed_e}")

                        # Add cropped base64 for each face
                        if face_analyses:
                            img = cv2.imread(image_path)
                            for i, face in enumerate(face_analyses):
                                region = face.get('region')
                                if region:
                                    cropped = img[region['y']:region['y'] + region['h'], region['x']:region['x'] + region['w']]
                                    _, buffer = cv2.imencode('.jpg', cropped)
                                    base64_str = base64.b64encode(buffer.tobytes()).decode('utf-8')
                                    face['cropped_base64'] = base64_str
                                    print(f"Added cropped base64 for face {i+1} in {filename}")

                        face_analyses = convert_to_serializable(face_analyses)
                    except Exception as e:
                        print(f"Error analyzing face for {filename}: {e}")
                        face_analyses = []

                    # Update database
                    try:
                        found = False
                        if 'image' in content and entity_idx == 1 and not is_frame:
                            content['face_analyses'] = face_analyses
                            content['last_processed'] = file_mtime
                            found = True
                        else:
                            entities_parent = content
                            if 'profileData' in content:
                                for key in content['profileData']:
                                    if 'entities' in content['profileData'][key]:
                                        entities_parent = content['profileData'][key]
                                        break
                            if 'entities' in entities_parent:
                                for entity in entities_parent['entities']:
                                    if entity.get('index') == entity_idx:
                                        if is_frame:
                                            if 'frame_face_analyses' not in entity:
                                                entity['frame_face_analyses'] = {}
                                            entity['frame_face_analyses'][frame_idx] = face_analyses
                                        else:
                                            entity['face_analyses'] = face_analyses
                                        entity['last_processed'] = file_mtime
                                        found = True
                                        break

                        if not found:
                            print(f"No place to add face analyses for node {node_id}, entity {entity_idx}")
                            continue

                        update_query = f'UPDATE "{project}_nodes" SET content = %s WHERE id = %s;'
                        cursor.execute(update_query, (json.dumps(content), node_id))
                        connection.commit()
                        print(f"Updated face analyses for node {node_id}, entity {entity_idx}" + (f", frame {frame_idx}" if is_frame else ""))
                        processed_files.add(filename)
                        any_new_processing = True  # Mark that we did new processing
                    except Exception as e:
                        print(f"Error updating face analyses for {filename}: {e}")

                except Exception as e:
                    print(f"Error accessing database for {filename}: {e}")

            # Generate visualizations - ALWAYS generate if there's any face data in the database
            # This ensures all analyzed faces are included in plots, not just newly processed ones
            print("\nGenerating face recognition distribution graphs...")
            all_face_analyses = []
            query = f'SELECT id, content FROM "{project}_nodes";'
            cursor.execute(query)
            rows = cursor.fetchall()

            for node_id, content in rows:
                if 'face_analyses' in content:
                    for face in content['face_analyses']:
                        all_face_analyses.append(face)
                entities_parent = content
                if 'profileData' in content:
                    for key in content['profileData']:
                        if 'entities' in content['profileData'][key]:
                            entities_parent = content['profileData'][key]
                            break
                if 'entities' in entities_parent:
                    for entity in entities_parent['entities']:
                        if 'face_analyses' in entity:
                            for face in entity['face_analyses']:
                                all_face_analyses.append(face)
                        if 'frame_face_analyses' in entity:
                            for frame_analyses in entity['frame_face_analyses'].values():
                                for face in frame_analyses:
                                    all_face_analyses.append(face)

            if all_face_analyses:
                graphs_dir = os.path.join(project_dir, 'graphs')
                os.makedirs(graphs_dir, exist_ok=True)
                visualization_paths = []

                print(f"Creating visualizations for {len(all_face_analyses)} total faces from database")

                # 1. Age distribution histogram
                ages = [face['age'] for face in all_face_analyses if 'age' in face]
                if ages:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(ages, bins=20, color='skyblue', edgecolor='black')
                    ax.set_title('Age Distribution Across Detected Faces')
                    ax.set_xlabel('Age')
                    ax.set_ylabel('Count')
                    age_path = os.path.join(graphs_dir, 'face_age_distribution.png')
                    plt.tight_layout()
                    plt.savefig(age_path, dpi=300)
                    plt.close()
                    visualization_paths.append(age_path)

                # 2. Gender distribution pie chart
                genders = [face['dominant_gender'] for face in all_face_analyses if 'dominant_gender' in face]
                if genders:
                    gender_counts = Counter(genders)
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.pie(gender_counts.values(), labels=gender_counts.keys(), autopct='%1.1f%%', startangle=90)
                    ax.set_title('Gender Distribution Across Detected Faces')
                    gender_path = os.path.join(graphs_dir, 'face_gender_distribution.png')
                    plt.savefig(gender_path, dpi=300)
                    plt.close()
                    visualization_paths.append(gender_path)

                # 3. Emotion distribution bar chart
                emotions = [face['dominant_emotion'] for face in all_face_analyses if 'dominant_emotion' in face]
                if emotions:
                    emotion_counts = Counter(emotions)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(emotion_counts.keys(), emotion_counts.values(), color='green', edgecolor='black')
                    ax.set_title('Emotion Distribution Across Detected Faces')
                    ax.set_xlabel('Emotion')
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    emotion_path = os.path.join(graphs_dir, 'face_emotion_distribution.png')
                    plt.tight_layout()
                    plt.savefig(emotion_path, dpi=300)
                    plt.close()
                    visualization_paths.append(emotion_path)

                # 4. Race distribution bar chart
                races = [face['dominant_race'] for face in all_face_analyses if 'dominant_race' in face]
                if races:
                    race_counts = Counter(races)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(race_counts.keys(), race_counts.values(), color='purple', edgecolor='black')
                    ax.set_title('Race Distribution Across Detected Faces')
                    ax.set_xlabel('Race')
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    race_path = os.path.join(graphs_dir, 'face_race_distribution.png')
                    plt.tight_layout()
                    plt.savefig(race_path, dpi=300)
                    plt.close()
                    visualization_paths.append(race_path)

                # 5. Gender confidence probability histogram
                gender_confs = [max(face['gender'].values()) for face in all_face_analyses if 'gender' in face and isinstance(face['gender'], dict)]
                if gender_confs:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(gender_confs, bins=20, color='blue', edgecolor='black')
                    ax.set_title('Gender Confidence Probability Distribution')
                    ax.set_xlabel('Confidence Score')
                    ax.set_ylabel('Count')
                    gender_conf_path = os.path.join(graphs_dir, 'face_gender_confidence.png')
                    plt.tight_layout()
                    plt.savefig(gender_conf_path, dpi=300)
                    plt.close()
                    visualization_paths.append(gender_conf_path)

                # 6. Emotion confidence probability histogram
                emotion_confs = [max(face['emotion'].values()) for face in all_face_analyses if 'emotion' in face and isinstance(face['emotion'], dict)]
                if emotion_confs:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(emotion_confs, bins=20, color='red', edgecolor='black')
                    ax.set_title('Emotion Confidence Probability Distribution')
                    ax.set_xlabel('Confidence Score')
                    ax.set_ylabel('Count')
                    emotion_conf_path = os.path.join(graphs_dir, 'face_emotion_confidence.png')
                    plt.tight_layout()
                    plt.savefig(emotion_conf_path, dpi=300)
                    plt.close()
                    visualization_paths.append(emotion_conf_path)

                # 7. Race confidence probability histogram
                race_confs = [max(face['race'].values()) for face in all_face_analyses if 'race' in face and isinstance(face['race'], dict)]
                if race_confs:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(race_confs, bins=20, color='orange', edgecolor='black')
                    ax.set_title('Race Confidence Probability Distribution')
                    ax.set_xlabel('Confidence Score')
                    ax.set_ylabel('Count')
                    race_conf_path = os.path.join(graphs_dir, 'face_race_confidence.png')
                    plt.tight_layout()
                    plt.savefig(race_conf_path, dpi=300)
                    plt.close()
                    visualization_paths.append(race_conf_path)

                # Send visualizations as base64
                print("\n=== Sending Face Recognition visualizations to endpoint ===")
                images = []
                for viz_path in visualization_paths:
                    if os.path.exists(viz_path):
                        with open(viz_path, 'rb') as f:
                            img_data = f.read()
                            base64_str = base64.b64encode(img_data).decode('utf-8')
                            images.append(base64_str)
                            print(f"Prepared visualization: {os.path.basename(viz_path)}")

                # Send the list of base64 images
                payload = {"images": images}
                try:
                    response = requests.post('http://127.0.0.1:4000/python-images', json=payload)
                    if response.status_code == 200:
                        print(f"Successfully sent {len(images)} visualizations")
                        if any_new_processing:
                            print("(Includes newly processed faces)")
                        else:
                            print("(All faces were previously processed, plots include existing data)")
                    else:
                        print(f"Failed to send visualizations: {response.status_code}, {response.text}")
                except Exception as e:
                    print(f"Error sending visualizations: {e}")

                print(f"\nGenerated {len(visualization_paths)} face recognition visualizations from {len(all_face_analyses)} total faces")
            else:
                print("No face analyses found in database for visualization")
    if 'face_recognition' in analyses_list and args.face_image_file:
        try:
            with open(args.face_image_file, 'r') as f:
                face_image_base64 = f.read().strip()
            print("Read base64 face image from file.")
            face_image_data = base64.b64decode(face_image_base64)
            face_image_array = np.frombuffer(face_image_data, np.uint8)
            face_image_cv = cv2.imdecode(face_image_array, cv2.IMREAD_COLOR)
            uploaded_face_objs = DeepFace.represent(face_image_cv, enforce_detection=True, detector_backend='opencv')
            if uploaded_face_objs:
                uploaded_face_embedding = np.array(uploaded_face_objs[0]['embedding'])
                print("Uploaded face embedding extracted successfully.")
            else:
                print("No face detected in uploaded image.")
                uploaded_face_embedding = None
                raise Exception("No face detected in uploaded image.")
            print("\nSearching for nodes with matching faces...")
            matches = []
            all_distances = []
            processed_nodes = set()  # Track processed node IDs
            select_query = f'SELECT id, content FROM "{project}_nodes";'
            cursor.execute(select_query)
            nodes = cursor.fetchall()
            for node_id, content in nodes:
                if node_id in processed_nodes:
                    print(f"Skipping already processed node {node_id}")
                    continue
                processed_nodes.add(node_id)  # Mark node as processed
                try:
                    content = json.loads(content) if isinstance(content, str) else content
                    if 'image' in content:
                        image_filename = get_image_filename(content['image'])
                        if image_filename:
                            image_path = os.path.join(image_folder, image_filename)
                            if os.path.isfile(image_path) and image_filename.lower().endswith((".jpg", ".jpeg", ".png")):
                                print(f"\nChecking image for node {node_id}: {image_filename}")
                                try:
                                    embeddings_objs = DeepFace.represent(image_path, enforce_detection=False)
                                    for i, emb_obj in enumerate(embeddings_objs):
                                        face_emb = np.array(emb_obj['embedding'])
                                        norm_face = np.linalg.norm(face_emb)
                                        norm_uploaded = np.linalg.norm(uploaded_face_embedding)
                                        dist = 1.0 if norm_face == 0 or norm_uploaded == 0 else ssd.cosine(face_emb / norm_face, uploaded_face_embedding / norm_uploaded)
                                        all_distances.append(dist)
                                        if dist < 0.68:
                                            match_info = f"Node {node_id}, top-level image (Face {i+1}, distance: {dist:.4f})"
                                            matches.append(match_info)
                                            print(f"Match found: {match_info}")
                                except Exception as e:
                                    print(f"Error processing image {image_filename} for node {node_id}: {e}")
                            else:
                                print(f"Skipping node {node_id}: Invalid or missing image filename")
                    entities_parent = content
                    if 'profileData' in content:
                        for key in content['profileData']:
                            if 'entities' in content['profileData'][key]:
                                entities_parent = content['profileData'][key]
                                break
                    if 'entities' in entities_parent:
                        for entity in entities_parent['entities']:
                            entity_idx = entity.get('index')
                            if not entity_idx:
                                continue
                            if 'image' in entity:
                                image_filename = get_image_filename(entity['image'])
                                if image_filename:
                                    image_path = os.path.join(image_folder, f"node_{node_id}_entity_{entity_idx}.jpg")
                                    if os.path.isfile(image_path) and image_filename.lower().endswith((".jpg", ".jpeg", ".png")):
                                        print(f"\nChecking image for node {node_id}, entity {entity_idx}: {image_filename}")
                                        try:
                                            embeddings_objs = DeepFace.represent(image_path, enforce_detection=False)
                                            for i, emb_obj in enumerate(embeddings_objs):
                                                face_emb = np.array(emb_obj['embedding'])
                                                norm_face = np.linalg.norm(face_emb)
                                                norm_uploaded = np.linalg.norm(uploaded_face_embedding)
                                                dist = 1.0 if norm_face == 0 or norm_uploaded == 0 else ssd.cosine(face_emb / norm_face, uploaded_face_embedding / norm_uploaded)
                                                all_distances.append(dist)
                                                if dist < 0.68:
                                                    match_info = f"Node {node_id}, entity {entity_idx} (Face {i+1}, distance: {dist:.4f})"
                                                    matches.append(match_info)
                                                    print(f"Match found: {match_info}")
                                        except Exception as e:
                                            print(f"Error processing image {image_filename} for node {node_id}, entity {entity_idx}: {e}")
                                    else:
                                        print(f"Skipping node {node_id}, entity {entity_idx}: Invalid or missing image filename")
                            if 'frame_face_analyses' in entity:
                                for frame_idx in entity['frame_face_analyses']:
                                    image_filename = f"node_{node_id}_entity_{entity_idx}_frame_{frame_idx}.jpg"
                                    image_path = os.path.join(image_folder, image_filename)
                                    if os.path.isfile(image_path):
                                        print(f"\nChecking frame image for node {node_id}, entity {entity_idx}, frame {frame_idx}: {image_filename}")
                                        try:
                                            embeddings_objs = DeepFace.represent(image_path, enforce_detection=False)
                                            for i, emb_obj in enumerate(embeddings_objs):
                                                face_emb = np.array(emb_obj['embedding'])
                                                norm_face = np.linalg.norm(face_emb)
                                                norm_uploaded = np.linalg.norm(uploaded_face_embedding)
                                                dist = 1.0 if norm_face == 0 or norm_uploaded == 0 else ssd.cosine(face_emb / norm_face, uploaded_face_embedding / norm_uploaded)
                                                all_distances.append(dist)
                                                if dist < 0.68:
                                                    match_info = f"Node {node_id}, entity {entity_idx}, frame {frame_idx} (Face {i+1}, distance: {dist:.4f})"
                                                    matches.append(match_info)
                                                    print(f"Match found: {match_info}")
                                        except Exception as e:
                                            print(f"Error processing frame image {image_filename}: {e}")
                except Exception as e:
                    print(f"Error in node loop: {e}")
            if matches:
                print("\nFound matching faces:")
                for m in matches:
                    print(m)
            else:
                print("\nNo matching faces found.")
            if all_distances:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(all_distances, bins=30, color='purple', alpha=0.7, edgecolor='black')
                ax.set_title('Distribution of Face Matching Distances')
                ax.set_xlabel('Cosine Distance')
                ax.set_ylabel('Frequency')
                ax.axvline(x=0.68, color='red', linestyle='--', label='Match Threshold (0.68)')
                ax.legend()
                dist_hist_path = os.path.join(graphs_dir, 'face_match_distances.png')
                plt.tight_layout()
                plt.savefig(dist_hist_path, dpi=300)
                plt.close()
                visualization_paths.append(dist_hist_path)
                # Send updated visualizations if any
                images = []
                for viz_path in visualization_paths:
                    with open(viz_path, 'rb') as f:
                        img_data = f.read()
                        base64_str = base64.b64encode(img_data).decode('utf-8')
                        images.append(base64_str)
                payload = {"images": images}
                try:
                    response = requests.post('http://127.0.0.1:4000/python-images', json=payload)
                    if response.status_code == 200:
                        print("Successfully sent visualization images.")
                    else:
                        print(f"Failed to send visualization images: {response.status_code}")
                except Exception as e:
                    print(f"Error sending visualization images: {e}")
        except Exception as e:
            print(f"Error in uploaded face matching: {e}")
    if 'face_clustering' in analyses_list:
        faces = []
        co_occurrence_map = defaultdict(lambda: defaultdict(int))  # Track which people appear together
        face_by_image = defaultdict(list)  # Group faces by their source image

        query = f"""SELECT id, content FROM "{project}_nodes";"""
        cursor.execute(query)
        rows = cursor.fetchall()

        for node_id, content in rows:
            entities_parents = [content]
            if 'profileData' in content:
                for key in content['profileData']:
                    if 'entities' in content['profileData'][key]:
                        entities_parents.append(content['profileData'][key])

            # Collect faces per image/frame
            image_faces = []

            for entities_parent in entities_parents:
                if 'entities' in entities_parent:
                    entities = entities_parent['entities']
                    for entity in entities:
                        entity_idx = entity.get('index')
                        if entity_idx is None:
                            continue

                        # Process static image faces
                        if 'face_analyses' in entity:
                            for face_idx, face in enumerate(entity['face_analyses']):
                                if 'embedding' in face:
                                    embedding = np.array(face['embedding'])
                                    if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                                        print(f"Skipping invalid embedding for node {node_id}, entity {entity_idx}, face {face_idx}")
                                        continue
                                    if np.all(embedding == 0):
                                        print(f"Skipping zero embedding for node {node_id}, entity {entity_idx}, face {face_idx}")
                                        continue

                                    face_data = (node_id, entity_idx, False, None, face_idx, embedding, face.copy())
                                    faces.append(face_data)
                                    image_faces.append(face_data)
                                    face_by_image[f"{node_id}_static_{entity_idx}"].append(face_data)

                        # Process video frame faces - group by frame
                        if 'frame_face_analyses' in entity:
                            for frame_idx_str in entity['frame_face_analyses']:
                                frame_idx = int(frame_idx_str)
                                analyses = entity['frame_face_analyses'][frame_idx_str]
                                frame_faces = []

                                for face_idx, face in enumerate(analyses):
                                    if 'embedding' in face:
                                        embedding = np.array(face['embedding'])
                                        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                                            print(f"Skipping invalid embedding for node {node_id}, entity {entity_idx}, frame {frame_idx}, face {face_idx}")
                                            continue
                                        if np.all(embedding == 0):
                                            print(f"Skipping zero embedding for node {node_id}, entity {entity_idx}, frame {frame_idx}, face {face_idx}")
                                            continue

                                        face_data = (node_id, entity_idx, True, frame_idx, face_idx, embedding, face.copy())
                                        faces.append(face_data)
                                        frame_faces.append(face_data)

                                # Group faces by frame
                                if frame_faces:
                                    face_by_image[f"{node_id}_frame_{entity_idx}_{frame_idx}"].extend(frame_faces)

                # Process faces at the parent level
                if 'face_analyses' in entities_parent:
                    for face_idx, face in enumerate(entities_parent['face_analyses']):
                        if 'embedding' in face:
                            embedding = np.array(face['embedding'])
                            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                                print(f"Skipping invalid embedding for node {node_id}, main image, face {face_idx}")
                                continue
                            if np.all(embedding == 0):
                                print(f"Skipping zero embedding for node {node_id}, main image, face {face_idx}")
                                continue

                            face_data = (node_id, 1, False, None, face_idx, embedding, face.copy())
                            faces.append(face_data)
                            face_by_image[f"{node_id}_main"].append(face_data)

        if not faces:
            print("No valid faces with embeddings found for clustering.")
        else:
            # First, identify unique persons across all images using face clustering
            embeddings = np.array([f[5] for f in faces])
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            valid_mask = norms.flatten() > 0
            embeddings = embeddings[valid_mask]
            faces = [faces[i] for i in range(len(faces)) if valid_mask[i]]

            if len(embeddings) < 2:
                print("Not enough valid embeddings for clustering (need at least 2).")
            else:
                # Normalize embeddings
                embeddings = embeddings / norms[valid_mask]

                # Compute distance matrix
                dist_condensed = ssd.pdist(embeddings, metric='cosine')
                if np.any(np.isnan(dist_condensed)) or np.any(np.isinf(dist_condensed)):
                    print("Warning: Distance matrix contains NaN or inf values. Replacing with maximum distance (1.0).")
                    dist_condensed = np.where(np.isfinite(dist_condensed), dist_condensed, 1.0)
                dist_matrix = ssd.squareform(dist_condensed)

                try:
                    # Perform hierarchical clustering to identify unique persons
                    linkage = sch.linkage(dist_condensed, method='average')
                    cluster_threshold = 0.68  # Threshold for same person
                    person_clusters = sch.fcluster(linkage, t=cluster_threshold, criterion='distance')

                    # Now analyze co-occurrences within same images/frames
                    print("\n=== Analyzing face co-occurrences in images ===")

                    # Build co-occurrence matrix
                    person_cooccurrence = defaultdict(lambda: defaultdict(int))
                    image_compositions = []  # Store which persons appear in each image

                    for image_key, image_faces in face_by_image.items():
                        if len(image_faces) > 1:  # Only consider images with multiple faces
                            # Get person IDs for faces in this image
                            persons_in_image = set()

                            for face_data in image_faces:
                                # Find this face in our global faces list to get its person cluster
                                for i, global_face in enumerate(faces):
                                    if (global_face[0] == face_data[0] and  # node_id
                                        global_face[1] == face_data[1] and  # entity_idx
                                        global_face[2] == face_data[2] and  # is_frame
                                        global_face[3] == face_data[3] and  # frame_idx
                                        global_face[4] == face_data[4]):    # face_idx
                                        person_id = int(person_clusters[i])  # Convert numpy.int32 to int
                                        persons_in_image.add(person_id)
                                        break

                            # Record co-occurrences
                            persons_list = list(persons_in_image)
                            if len(persons_list) > 1:
                                for i in range(len(persons_list)):
                                    for j in range(i + 1, len(persons_list)):
                                        person_cooccurrence[persons_list[i]][persons_list[j]] += 1
                                        person_cooccurrence[persons_list[j]][persons_list[i]] += 1

                                image_compositions.append({
                                    'image_key': image_key,
                                    'persons': persons_list,
                                    'num_faces': len(image_faces)
                                })

                    # Update cluster IDs in database with person identifiers
                    cluster_map = defaultdict(list)
                    for i, cluster_id in enumerate(person_clusters):
                        node_id, entity_idx, is_frame, frame_idx, face_idx = faces[i][:5]
                        # Convert numpy.int32 to regular Python int
                        cluster_map[node_id].append((entity_idx, is_frame, frame_idx, face_idx, int(cluster_id)))

                    # Update database with person clusters
                    for node_id, updates in cluster_map.items():
                        try:
                            select_query = f'SELECT content FROM "{project}_nodes" WHERE id = %s;'
                            cursor.execute(select_query, (node_id,))
                            row = cursor.fetchone()
                            if not row:
                                continue
                            content = row[0]

                            entities_parents = [content]
                            if 'profileData' in content:
                                for key in content['profileData']:
                                    if 'entities' in content['profileData'][key]:
                                        entities_parents.append(content['profileData'][key])

                            for entity_idx, is_frame, frame_idx, face_idx, cluster_id in updates:
                                found = False
                                for entities_parent in entities_parents:
                                    if 'entities' in entities_parent:
                                        entities = entities_parent['entities']
                                        for entity in entities:
                                            if entity.get('index') == entity_idx:
                                                if is_frame:
                                                    if 'frame_face_analyses' in entity and str(frame_idx) in entity['frame_face_analyses']:
                                                        analyses = entity['frame_face_analyses'][str(frame_idx)]
                                                        if face_idx < len(analyses):
                                                            analyses[face_idx]['person_id'] = int(cluster_id)
                                                            found = True
                                                else:
                                                    if 'face_analyses' in entity:
                                                        analyses = entity['face_analyses']
                                                        if face_idx < len(analyses):
                                                            analyses[face_idx]['person_id'] = int(cluster_id)
                                                            found = True
                                                break
                                    if not found and not is_frame and entity_idx == 1:
                                        if 'face_analyses' in entities_parent:
                                            analyses = entities_parent['face_analyses']
                                            if face_idx < len(analyses):
                                                analyses[face_idx]['person_id'] = int(cluster_id)
                                                found = True

                            # Add co-occurrence data to content
                            content['face_cooccurrences'] = {}
                            for person_id in set([u[4] for u in updates]):
                                if person_id in person_cooccurrence:
                                    # Convert numpy.int32 keys to strings for JSON serialization
                                    content['face_cooccurrences'][str(int(person_id))] = {
                                        str(int(k)): int(v) for k, v in person_cooccurrence[person_id].items()
                                    }

                            update_query = f'UPDATE "{project}_nodes" SET content = %s WHERE id = %s;'
                            cursor.execute(update_query, (json.dumps(content), node_id))
                            connection.commit()
                        except Exception as e:
                            print(f"Error updating person_ids for node {node_id}: {e}")

                    # Create relationship graph based on co-occurrences
                    print("\n=== Creating person relationship network ===")

                    # Create tables for person relationships
                    try:
                        cursor.execute(f'DROP TABLE IF EXISTS "{project}_person_nodes";')
                        cursor.execute(f'DROP TABLE IF EXISTS "{project}_person_edges";')
                        cursor.execute(f'''
                            CREATE TABLE "{project}_person_nodes" (
                                id SERIAL PRIMARY KEY,
                                person_id INTEGER UNIQUE,
                                face_count INTEGER,
                                image_count INTEGER,
                                content JSONB
                            );
                        ''')
                        cursor.execute(f'''
                            CREATE TABLE "{project}_person_edges" (
                                from_person INTEGER,
                                to_person INTEGER,
                                weight INTEGER,
                                PRIMARY KEY (from_person, to_person)
                            );
                        ''')
                        connection.commit()
                    except Exception as e:
                        print(f"Error creating person tables: {e}")
                        connection.rollback()

                    # Insert person nodes
                    person_stats = defaultdict(lambda: {'face_count': 0, 'image_count': 0, 'images': set()})
                    for i, person_id in enumerate(person_clusters):
                        person_stats[int(person_id)]['face_count'] += 1

                    for image_comp in image_compositions:
                        for person_id in image_comp['persons']:
                            person_stats[int(person_id)]['images'].add(image_comp['image_key'])
                            person_stats[int(person_id)]['image_count'] = len(person_stats[int(person_id)]['images'])

                    for person_id, stats in person_stats.items():
                        content = {
                            'person_id': int(person_id),
                            'face_count': stats['face_count'],
                            'image_count': stats['image_count']
                        }
                        insert_query = f'''
                            INSERT INTO "{project}_person_nodes" (person_id, face_count, image_count, content)
                            VALUES (%s, %s, %s, %s);
                        '''
                        cursor.execute(insert_query, (
                            int(person_id),
                            stats['face_count'],
                            stats['image_count'],
                            json.dumps(content)
                        ))

                    # Insert person edges (relationships based on co-occurrence)
                    edges_to_insert = []
                    for person1 in person_cooccurrence:
                        for person2, weight in person_cooccurrence[person1].items():
                            if person1 < person2:  # Avoid duplicates
                                edges_to_insert.append((int(person1), int(person2), weight))

                    if edges_to_insert:
                        insert_edge_query = f'''
                            INSERT INTO "{project}_person_edges" (from_person, to_person, weight)
                            VALUES (%s, %s, %s);
                        '''
                        cursor.executemany(insert_edge_query, edges_to_insert)

                    connection.commit()
                    print(f"Created person relationship network with {len(person_stats)} persons and {len(edges_to_insert)} relationships")

                    # Generate visualizations for person relationships
                    visualization_paths = []
                    graphs_dir = os.path.join(project_dir, 'graphs')
                    os.makedirs(graphs_dir, exist_ok=True)

                    # 1. Co-occurrence heatmap
                    if person_cooccurrence:
                        persons = sorted(set(list(person_cooccurrence.keys()) +
                                        [p for subdict in person_cooccurrence.values() for p in subdict.keys()]))
                        matrix = np.zeros((len(persons), len(persons)))
                        person_to_idx = {p: i for i, p in enumerate(persons)}

                        for p1 in person_cooccurrence:
                            for p2, count in person_cooccurrence[p1].items():
                                matrix[person_to_idx[p1]][person_to_idx[p2]] = count

                        fig, ax = plt.subplots(figsize=(12, 10))
                        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
                        ax.set_xticks(range(len(persons)))
                        ax.set_yticks(range(len(persons)))
                        ax.set_xticklabels([f"Person {p}" for p in persons], rotation=45, ha='right')
                        ax.set_yticklabels([f"Person {p}" for p in persons])
                        ax.set_title('Person Co-occurrence Matrix (Who appears with whom)')
                        plt.colorbar(im, ax=ax, label='Number of images together')
                        plt.tight_layout()

                        heatmap_path = os.path.join(graphs_dir, 'person_cooccurrence_heatmap.png')
                        plt.savefig(heatmap_path)
                        plt.close()
                        visualization_paths.append(heatmap_path)

                    # 2. Network graph of relationships
                    if edges_to_insert:
                        G = nx.Graph()
                        for person_id, stats in person_stats.items():
                            G.add_node(person_id,
                                    face_count=stats['face_count'],
                                    image_count=stats['image_count'])

                        for p1, p2, weight in edges_to_insert:
                            G.add_edge(p1, p2, weight=weight)

                        fig, ax = plt.subplots(figsize=(14, 10))
                        pos = nx.spring_layout(G, k=2, iterations=50)

                        # Node sizes based on face count
                        node_sizes = [person_stats[node]['face_count'] * 100 for node in G.nodes()]

                        # Edge widths based on co-occurrence count
                        edge_widths = [G[u][v]['weight'] * 2 for u, v in G.edges()]

                        # Draw network
                        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                                            node_color='lightblue', alpha=0.7, ax=ax)
                        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, ax=ax)
                        nx.draw_networkx_labels(G, pos,
                                            labels={n: f"P{n}" for n in G.nodes()},
                                            font_size=10, ax=ax)

                        ax.set_title('Person Relationship Network (Based on Co-appearances)', fontsize=16)
                        ax.axis('off')

                        network_path = os.path.join(graphs_dir, 'person_relationship_network.png')
                        plt.savefig(network_path)
                        plt.close()
                        visualization_paths.append(network_path)

                    # 3. Bar chart of most connected persons
                    if person_cooccurrence:
                        person_connections = {}
                        for person_id in person_stats:
                            connections = sum(person_cooccurrence.get(person_id, {}).values())
                            person_connections[person_id] = connections

                        # Sort by connections and take top 20
                        top_persons = sorted(person_connections.items(),
                                        key=lambda x: x[1], reverse=True)[:20]

                        if top_persons:
                            fig, ax = plt.subplots(figsize=(12, 6))
                            persons = [f"Person {p[0]}" for p in top_persons]
                            connections = [p[1] for p in top_persons]

                            bars = ax.bar(range(len(persons)), connections, color='steelblue')
                            ax.set_xticks(range(len(persons)))
                            ax.set_xticklabels(persons, rotation=45, ha='right')
                            ax.set_ylabel('Total Co-occurrences')
                            ax.set_title('Most Connected Persons (Appearing with Others)')

                            # Add value labels on bars
                            for bar, val in zip(bars, connections):
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{int(val)}', ha='center', va='bottom')

                            plt.tight_layout()
                            bar_path = os.path.join(graphs_dir, 'person_connections_bar.png')
                            plt.savefig(bar_path)
                            plt.close()
                            visualization_paths.append(bar_path)

                    # 4. Image composition analysis
                    if image_compositions:
                        # Distribution of number of persons per image
                        persons_per_image = [len(comp['persons']) for comp in image_compositions]

                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                        # Histogram
                        ax1.hist(persons_per_image, bins=range(2, max(persons_per_image) + 2),
                                edgecolor='black', alpha=0.7, color='green')
                        ax1.set_xlabel('Number of Different Persons in Image')
                        ax1.set_ylabel('Number of Images')
                        ax1.set_title('Distribution of Group Sizes in Images')
                        ax1.set_xticks(range(2, max(persons_per_image) + 1))

                        # Pie chart
                        group_sizes = Counter(persons_per_image)
                        ax2.pie(group_sizes.values(), labels=[f"{k} persons" for k in group_sizes.keys()],
                            autopct='%1.1f%%', startangle=90)
                        ax2.set_title('Proportion of Group Sizes')

                        plt.tight_layout()
                        composition_path = os.path.join(graphs_dir, 'image_composition_analysis.png')
                        plt.savefig(composition_path)
                        plt.close()
                        visualization_paths.append(composition_path)

                    # Send visualizations to endpoint
                    print("\n=== Sending Person Relationship visualizations to endpoint ===")
                    images = []
                    for viz_path in visualization_paths:
                        if os.path.exists(viz_path):
                            with open(viz_path, 'rb') as f:
                                img_data = f.read()
                                base64_str = base64.b64encode(img_data).decode('utf-8')
                                images.append(base64_str)
                                print(f"Prepared visualization: {os.path.basename(viz_path)}")

                    if images:
                        payload = {"images": images}
                        try:
                            response = requests.post('http://127.0.0.1:4000/python-images', json=payload)
                            if response.status_code == 200:
                                print(f"Successfully sent {len(images)} visualizations")
                            else:
                                print(f"Failed to send visualizations: {response.status_code}, {response.text}")
                        except Exception as e:
                            print(f"Error sending visualizations: {e}")

                    print(f"\nGenerated {len(visualization_paths)} person relationship visualizations")

                    # Send graph data for person co-occurrence visualization
                    nodes = []
                    edges = []

                    for person_id, stats in person_stats.items():
                        nodes.append({
                            "id": f"person_{person_id}",
                            "label": f"Person {person_id}",
                            "group": "person",
                            "face_count": stats['face_count'],
                            "image_count": stats['image_count']
                        })

                    for p1, p2, weight in edges_to_insert:
                        edges.append({
                            "from": f"person_{p1}",
                            "to": f"person_{p2}",
                            "weight": weight,
                            "label": f"{weight} images together"
                        })

                    graph_data = {
                        "type": "graph_data",
                        "nodes": nodes,
                        "edges": edges,
                        "description": "Person co-occurrence network showing who appears together in images"
                    }

                    try:
                        response = requests.post('http://127.0.0.1:4000/graphdata', json=graph_data)
                        if response.status_code == 200:
                            print("Person co-occurrence graph data sent successfully")
                        else:
                            print(f"Error sending co-occurrence graph data: {response.status_code}")
                    except Exception as e:
                        print(f"Error sending co-occurrence graph data: {e}")

                    print("\n=== Computing centralities for person co-occurrence network ===")

                    if edges_to_insert and len(person_stats) > 1:
                        # Create NetworkX graph from person co-occurrences
                        G_person = nx.Graph()  # Undirected since co-occurrence is symmetric

                        for person_id, stats in person_stats.items():
                            G_person.add_node(person_id,
                                            face_count=stats['face_count'],
                                            image_count=stats['image_count'])

                        for p1, p2, weight in edges_to_insert:
                            G_person.add_edge(p1, p2, weight=weight)

                        print(f"Person co-occurrence network has {len(G_person.nodes)} persons and {len(G_person.edges)} relationships")

                        # Compute centrality measures (adapted from face network analysis)
                        person_betweenness = nx.betweenness_centrality(G_person, weight='weight')
                        person_closeness = nx.closeness_centrality(G_person, distance='weight')
                        person_degree = nx.degree_centrality(G_person)
                        person_clustering = nx.clustering(G_person, weight='weight')

                        # For undirected graph, in_degree and out_degree are the same as degree
                        person_in_degree = person_degree  # Same for undirected
                        person_out_degree = person_degree  # Same for undirected

                        # Eigenvector centrality
                        try:
                            person_eigenvector = nx.eigenvector_centrality(G_person, weight='weight', max_iter=50000, tol=1e-03)
                        except nx.PowerIterationFailedConvergence:
                            print("Eigenvector centrality failed to converge. Setting to degree centrality as fallback.")
                            person_eigenvector = person_degree

                        # Katz centrality (for undirected graph)
                        try:
                            # Calculate appropriate alpha for Katz centrality
                            A = nx.to_numpy_array(G_person)
                            eigenvalues = np.linalg.eigvals(A)
                            lambda_max = np.max(np.abs(eigenvalues.real))
                            alpha = 0.85 / (lambda_max + 1e-6) if lambda_max > 0 else 0.01
                            person_katz = nx.katz_centrality(G_person, alpha=alpha, max_iter=10000, tol=1e-03, weight='weight')
                        except Exception as e:
                            print(f"Katz centrality computation failed: {e}. Using fallback alpha.")
                            try:
                                person_katz = nx.katz_centrality(G_person, alpha=0.01, max_iter=10000, tol=1e-03, weight='weight')
                            except:
                                print("Katz centrality failed completely. Setting all to 0.")
                                person_katz = {node: 0.0 for node in G_person.nodes}

                        # Update person_nodes table with centralities
                        for person_id in G_person.nodes:
                            centralities = {
                                'face_betweenness': person_betweenness.get(person_id, 0),
                                'face_closeness': person_closeness.get(person_id, 0),
                                'face_in_degree': person_in_degree.get(person_id, 0),
                                'face_out_degree': person_out_degree.get(person_id, 0),
                                'face_eigenvector': person_eigenvector.get(person_id, 0),
                                'face_cluster': person_clustering.get(person_id, 0),
                                'face_katz': person_katz.get(person_id, 0)
                            }

                            # Update the person_nodes table with centralities
                            update_query = f'''
                                UPDATE "{project}_person_nodes"
                                SET content = content || %s
                                WHERE person_id = %s;
                            '''
                            cursor.execute(update_query, (json.dumps({'centralities': centralities}), int(person_id)))

                        connection.commit()
                        print("Person centralities computed and stored in person_nodes table")

                        # Update original nodes table with person centralities
                        print("Updating original nodes with person-based centralities...")

                        # Map person centralities back to original nodes
                        for node_id, updates in cluster_map.items():
                            try:
                                # Get all person IDs associated with this node
                                person_ids_in_node = set([int(u[4]) for u in updates])

                                # Aggregate centralities if multiple persons in the node
                                aggregated_centralities = {
                                    'person_betweenness': 0,
                                    'person_closeness': 0,
                                    'person_in_degree': 0,
                                    'person_out_degree': 0,
                                    'person_eigenvector': 0,
                                    'person_cluster': 0,
                                    'person_katz': 0
                                }

                                # Average the centralities of all persons in this node
                                if person_ids_in_node:
                                    for person_id in person_ids_in_node:
                                        if person_id in G_person.nodes:
                                            aggregated_centralities['person_betweenness'] += person_betweenness.get(person_id, 0)
                                            aggregated_centralities['person_closeness'] += person_closeness.get(person_id, 0)
                                            aggregated_centralities['person_in_degree'] += person_in_degree.get(person_id, 0)
                                            aggregated_centralities['person_out_degree'] += person_out_degree.get(person_id, 0)
                                            aggregated_centralities['person_eigenvector'] += person_eigenvector.get(person_id, 0)
                                            aggregated_centralities['person_cluster'] += person_clustering.get(person_id, 0)
                                            aggregated_centralities['person_katz'] += person_katz.get(person_id, 0)

                                    # Average the values
                                    num_persons = len(person_ids_in_node)
                                    for key in aggregated_centralities:
                                        aggregated_centralities[key] /= num_persons

                                # Update the original nodes table
                                cursor.execute(f'SELECT content FROM "{project}_nodes" WHERE id = %s;', (node_id,))
                                row = cursor.fetchone()
                                if row:
                                    content = row[0]

                                    # Add or update the centralities field
                                    if 'person_centralities' not in content:
                                        content['person_centralities'] = {}

                                    # Add person-based centralities to existing centralities
                                    content['person_centralities'].update(aggregated_centralities)

                                    # Update the database
                                    cursor.execute(f'UPDATE "{project}_nodes" SET content = %s WHERE id = %s;',
                                                (json.dumps(content), node_id))
                                    connection.commit()

                            except Exception as e:
                                print(f"Error updating person centralities for node {node_id}: {e}")

                        print(f"Updated {len(cluster_map)} original nodes with person-based centralities")

                        # Generate centrality visualizations
                        centrality_images = []
                        graphs_dir = os.path.join(project_dir, 'graphs')
                        os.makedirs(graphs_dir, exist_ok=True)

                        # 1. Histogram of centralities
                        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
                        axes = axes.flatten()

                        centrality_measures = [
                            ('face_betweenness', person_betweenness, 'Betweenness Centrality'),
                            ('face_closeness', person_closeness, 'Closeness Centrality'),
                            ('face_in_degree', person_in_degree, 'In-Degree Centrality'),
                            ('face_out_degree', person_out_degree, 'Out-Degree Centrality'),
                            ('face_eigenvector', person_eigenvector, 'Eigenvector Centrality'),
                            ('face_cluster', person_clustering, 'Clustering Coefficient'),
                            ('face_katz', person_katz, 'Katz Centrality')
                        ]

                        for idx, (ax, (name, values_dict, title)) in enumerate(zip(axes[:7], centrality_measures)):
                            values = list(values_dict.values())
                            if values:
                                # Create histogram
                                hist, bins = np.histogram(values, bins=20)
                                hist_pct = hist * 100 / np.sum(hist) if np.sum(hist) > 0 else hist

                                ax.bar(bins[:-1], hist_pct, width=np.diff(bins),
                                    edgecolor='black', align='edge', alpha=0.7,
                                    color=plt.get_cmap("Paired")(idx % 12))
                                ax.set_title(title)
                                ax.set_xlabel('Centrality Value')
                                ax.set_ylabel('Percentage of Persons [%]')
                                ax.set_yscale('log')

                        # Hide unused subplots
                        for ax in axes[7:]:
                            ax.axis('off')

                        plt.suptitle('Person Co-occurrence Network Centrality Distributions', fontsize=16)
                        plt.tight_layout()

                        centrality_hist_path = os.path.join(graphs_dir, 'person_centralities_hist.png')
                        plt.savefig(centrality_hist_path, dpi=150)
                        plt.close()

                        with open(centrality_hist_path, 'rb') as f:
                            img_data = f.read()
                            base64_str = base64.b64encode(img_data).decode('utf-8')
                            centrality_images.append(base64_str)

                        # 2. Top persons by each centrality measure
                        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
                        axes = axes.flatten()

                        for idx, (ax, (name, values_dict, title)) in enumerate(zip(axes[:7], centrality_measures)):
                            # Get top 10 persons for this centrality
                            sorted_persons = sorted(values_dict.items(), key=lambda x: x[1], reverse=True)[:10]

                            if sorted_persons:
                                persons = [f"P{p[0]}" for p in sorted_persons]
                                values = [p[1] for p in sorted_persons]

                                bars = ax.barh(range(len(persons)), values, color=plt.get_cmap("Paired")(idx % 12))
                                ax.set_yticks(range(len(persons)))
                                ax.set_yticklabels(persons)
                                ax.set_xlabel('Centrality Value')
                                ax.set_title(f'Top 10 Persons by {title}')
                                ax.invert_yaxis()  # Highest at top

                                # Add value labels
                                for bar, val in zip(bars, values):
                                    width = bar.get_width()
                                    ax.text(width, bar.get_y() + bar.get_height()/2.,
                                        f'{val:.3f}', ha='left', va='center', fontsize=8)

                        # Hide unused subplots
                        for ax in axes[7:]:
                            ax.axis('off')

                        plt.suptitle('Top Persons by Centrality Measures', fontsize=16)
                        plt.tight_layout()

                        top_persons_path = os.path.join(graphs_dir, 'person_top_centralities.png')
                        plt.savefig(top_persons_path, dpi=150)
                        plt.close()

                        with open(top_persons_path, 'rb') as f:
                            img_data = f.read()
                            base64_str = base64.b64encode(img_data).decode('utf-8')
                            centrality_images.append(base64_str)

                        # 3. Network visualization with node sizes based on centrality
                        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                        axes = axes.flatten()

                        # Use spring layout for all visualizations
                        pos = nx.spring_layout(G_person, k=2, iterations=50, seed=42)

                        for idx, (ax, (name, values_dict, title)) in enumerate(zip(axes[:7], centrality_measures)):
                            # Node sizes based on centrality value
                            node_sizes = [values_dict.get(node, 0) * 1000 + 50 for node in G_person.nodes()]

                            # Edge widths based on weight
                            edge_widths = [G_person[u][v]['weight'] * 0.5 for u, v in G_person.edges()]

                            # Draw network
                            nx.draw_networkx_nodes(G_person, pos, node_size=node_sizes,
                                                node_color=list(values_dict.values()),
                                                cmap='YlOrRd', alpha=0.7, ax=ax)
                            nx.draw_networkx_edges(G_person, pos, width=edge_widths,
                                                alpha=0.3, ax=ax)
                            nx.draw_networkx_labels(G_person, pos,
                                                labels={n: f"P{n}" for n in G_person.nodes()},
                                                font_size=8, ax=ax)

                            ax.set_title(f'Network colored by {title}', fontsize=10)
                            ax.axis('off')

                        # Hide unused subplot
                        axes[7].axis('off')

                        plt.suptitle('Person Network Visualizations by Centrality', fontsize=16)
                        plt.tight_layout()

                        network_centrality_path = os.path.join(graphs_dir, 'person_network_centralities.png')
                        plt.savefig(network_centrality_path, dpi=150)
                        plt.close()

                        with open(network_centrality_path, 'rb') as f:
                            img_data = f.read()
                            base64_str = base64.b64encode(img_data).decode('utf-8')
                            centrality_images.append(base64_str)

                        # 4. Correlation matrix between centrality measures
                        centrality_data = []
                        for person_id in G_person.nodes:
                            centrality_data.append({
                                'person_id': person_id,
                                'betweenness': person_betweenness.get(person_id, 0),
                                'closeness': person_closeness.get(person_id, 0),
                                'degree': person_degree.get(person_id, 0),
                                'eigenvector': person_eigenvector.get(person_id, 0),
                                'clustering': person_clustering.get(person_id, 0),
                                'katz': person_katz.get(person_id, 0)
                            })

                        if centrality_data:
                            df_centralities = pd.DataFrame(centrality_data)

                            # Compute correlation matrix
                            corr_matrix = df_centralities.drop('person_id', axis=1).corr()

                            fig, ax = plt.subplots(figsize=(10, 8))
                            im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

                            # Set ticks and labels
                            centrality_names = ['Betweenness', 'Closeness', 'Degree', 'Eigenvector', 'Clustering', 'Katz']
                            ax.set_xticks(range(len(centrality_names)))
                            ax.set_yticks(range(len(centrality_names)))
                            ax.set_xticklabels(centrality_names, rotation=45, ha='right')
                            ax.set_yticklabels(centrality_names)

                            # Add correlation values
                            for i in range(len(centrality_names)):
                                for j in range(len(centrality_names)):
                                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                                ha='center', va='center', color='black' if abs(corr_matrix.iloc[i, j]) < 0.5 else 'white')

                            ax.set_title('Correlation Matrix of Person Network Centralities')
                            plt.colorbar(im, ax=ax, label='Correlation')
                            plt.tight_layout()

                            corr_matrix_path = os.path.join(graphs_dir, 'person_centrality_correlation.png')
                            plt.savefig(corr_matrix_path, dpi=150)
                            plt.close()

                            with open(corr_matrix_path, 'rb') as f:
                                img_data = f.read()
                                base64_str = base64.b64encode(img_data).decode('utf-8')
                                centrality_images.append(base64_str)

                        # Send centrality visualizations
                        if centrality_images:
                            payload = {"images": centrality_images}
                            try:
                                response = requests.post('http://127.0.0.1:4000/python-images', json=payload)
                                if response.status_code == 200:
                                    print(f"Successfully sent {len(centrality_images)} person centrality visualizations")
                                else:
                                    print(f"Failed to send centrality visualizations: {response.status_code}")
                            except Exception as e:
                                print(f"Error sending centrality visualizations: {e}")

                        print(f"Person network analysis completed with {len(centrality_measures)} centrality measures")

                    else:
                        print("Not enough persons or relationships for centrality analysis")
                        # Set default centralities for all persons
                        for person_id in person_stats:
                            centralities = {
                                'face_betweenness': 0,
                                'face_closeness': 0,
                                'face_in_degree': 0,
                                'face_out_degree': 0,
                                'face_eigenvector': 0,
                                'face_cluster': 0,
                                'face_katz': 0
                            }

                            update_query = f'''
                                UPDATE "{project}_person_nodes"
                                SET content = content || %s
                                WHERE person_id = %s;
                            '''
                            cursor.execute(update_query, (json.dumps({'centralities': centralities}), int(person_id)))

                        connection.commit()
                        print("Set default centralities for all persons due to insufficient network data")

                except Exception as cluster_e:
                    print(f"Error during clustering: {cluster_e}")

    connection.close()

if __name__ == "__main__":
    main()
