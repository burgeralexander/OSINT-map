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
        # Character-based chunking parameters
        char_chunk_size = 100000  # Approximate safe character length (~2000-3000 tokens)
        char_chunk_overlap = 200  # Overlap in characters
        # Define optional analyses
        enabled_analyses = ['language', 'sentiment', 'ner']
        # Storage for analysis results
        all_nlp_results = []
        sentiment_scores = []
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

        # Visualization section (unchanged)
        print("\n=== Creating NLP Visualizations ===")
        graphs_dir = os.path.join(project_dir, 'graphs')
        os.makedirs(graphs_dir, exist_ok=True)
        visualization_paths = []
        # [Rest of the visualization section remains unchanged]
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
                        print(f"Face analyses exist and file unchanged for node {node_id}, entity {entity_idx}, skipping")
                        skip_processing = True
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
                                        print(f"Frame face analyses exist and file unchanged for node {node_id}, entity {entity_idx}, frame {frame_idx}, skipping")
                                        skip_processing = True
                                else:
                                    if 'face_analyses' in entity and entity.get('last_processed', 0) >= file_mtime:
                                        print(f"Face analyses exist and file unchanged for node {node_id}, entity {entity_idx}, skipping")
                                        skip_processing = True
                                break

                if not found_place:
                    print(f"No place found to add face analyses for {filename}, skipping")
                    continue

                if skip_processing:
                    processed_files.add(filename)
                    continue

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
                except Exception as e:
                    print(f"Error updating face analyses for {filename}: {e}")

            except Exception as e:
                print(f"Error accessing database for {filename}: {e}")

        # Generate visualizations only if new analyses were added
        if processed_files:
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
                    else:
                        print(f"Failed to send visualizations: {response.status_code}, {response.text}")
                except Exception as e:
                    print(f"Error sending visualizations: {e}")

                print(f"\nGenerated {len(visualization_paths)} face recognition visualizations")
            else:
                print("No face analyses found for visualization")
        else:
            print("No new images processed, skipping visualization generation")
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
        query = f"""SELECT id, content FROM "{project}_nodes";"""
        cursor.execute(query)
        rows = cursor.fetchall()
        for node_id, content in rows:
            entities_parents = [content]
            if 'profileData' in content:
                for key in content['profileData']:
                    if 'entities' in content['profileData'][key]:
                        entities_parents.append(content['profileData'][key])
            for entities_parent in entities_parents:
                if 'entities' in entities_parent:
                    entities = entities_parent['entities']
                    for entity in entities:
                        entity_idx = entity.get('index')
                        if entity_idx is None:
                            continue
                        if 'face_analyses' in entity:
                            for face_idx, face in enumerate(entity['face_analyses']):
                                if 'embedding' in face:
                                    embedding = np.array(face['embedding'])
                                    if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                                        print(f"Skipping invalid embedding (NaN or inf) for node {node_id}, entity {entity_idx}, face {face_idx}")
                                        continue
                                    if np.all(embedding == 0):
                                        print(f"Skipping zero embedding for node {node_id}, entity {entity_idx}, face {face_idx}")
                                        continue
                                    faces.append((node_id, entity_idx, False, None, face_idx, embedding, face.copy()))
                        if 'frame_face_analyses' in entity:
                            for frame_idx_str in entity['frame_face_analyses']:
                                frame_idx = int(frame_idx_str)
                                analyses = entity['frame_face_analyses'][frame_idx_str]
                                for face_idx, face in enumerate(analyses):
                                    if 'embedding' in face:
                                        embedding = np.array(face['embedding'])
                                        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                                            print(f"Skipping invalid embedding (NaN or inf) for node {node_id}, entity {entity_idx}, frame {frame_idx}, face {face_idx}")
                                            continue
                                        if np.all(embedding == 0):
                                            print(f"Skipping zero embedding for node {node_id}, entity {entity_idx}, frame {frame_idx}, face {face_idx}")
                                            continue
                                        faces.append((node_id, entity_idx, True, frame_idx, face_idx, embedding, face.copy()))
                if 'face_analyses' in entities_parent:
                    for face_idx, face in enumerate(entities_parent['face_analyses']):
                        if 'embedding' in face:
                            embedding = np.array(face['embedding'])
                            if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
                                print(f"Skipping invalid embedding (NaN or inf) for node {node_id}, main image, face {face_idx}")
                                continue
                            if np.all(embedding == 0):
                                print(f"Skipping zero embedding for node {node_id}, main image, face {face_idx}")
                                continue
                            faces.append((node_id, 1, False, None, face_idx, embedding, face.copy()))
        if not faces:
            print("No valid faces with embeddings found for clustering.")
        else:
            embeddings = np.array([f[5] for f in faces])
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            valid_mask = norms.flatten() > 0
            embeddings = embeddings[valid_mask]
            faces = [faces[i] for i in range(len(faces)) if valid_mask[i]]
            if len(embeddings) < 2:
                print("Not enough valid embeddings for clustering (need at least 2).")
            else:
                embeddings = embeddings / norms[valid_mask]
                dist_condensed = ssd.pdist(embeddings, metric='cosine')
                if np.any(np.isnan(dist_condensed)) or np.any(np.isinf(dist_condensed)):
                    print("Warning: Distance matrix contains NaN or inf values. Replacing with maximum distance (1.0).")
                    dist_condensed = np.where(np.isfinite(dist_condensed), dist_condensed, 1.0)
                dist_matrix = ssd.squareform(dist_condensed)
                try:
                    linkage = sch.linkage(dist_condensed, method='average')
                    cluster_threshold = 0.68
                    clusters = sch.fcluster(linkage, t=cluster_threshold, criterion='distance')
                    cluster_map = defaultdict(list)
                    for i, cluster_id in enumerate(clusters):
                        node_id, entity_idx, is_frame, frame_idx, face_idx = faces[i][:5]
                        cluster_map[node_id].append((entity_idx, is_frame, frame_idx, face_idx, cluster_id))
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
                                                            analyses[face_idx]['cluster_id'] = int(cluster_id)
                                                            found = True
                                                else:
                                                    if 'face_analyses' in entity:
                                                        analyses = entity['face_analyses']
                                                        if face_idx < len(analyses):
                                                            analyses[face_idx]['cluster_id'] = int(cluster_id)
                                                            found = True
                                                break
                                    if not found and not is_frame and entity_idx == 1:
                                        if 'face_analyses' in entities_parent:
                                            analyses = entities_parent['face_analyses']
                                            if face_idx < len(analyses):
                                                analyses[face_idx]['cluster_id'] = int(cluster_id)
                                                found = True
                                if not found:
                                    print(f"Could not update cluster_id for node {node_id}, entity {entity_idx}, frame {frame_idx}, face {face_idx}")
                            update_query = f'UPDATE "{project}_nodes" SET content = %s WHERE id = %s;'
                            cursor.execute(update_query, (json.dumps(content), node_id))
                            connection.commit()
                            #print(f"Updated cluster_ids for node {node_id}")
                        except Exception as e:
                            print(f"Error updating cluster_ids for node {node_id}: {e}")
                    # Create face_nodes and face_edges tables for face network analysis
                    # Create face tables (drop first to ensure clean schema)
                    try:
                        cursor.execute(f'DROP TABLE IF EXISTS "{project}_face_nodes";')
                        cursor.execute(f'DROP TABLE IF EXISTS "{project}_face_edges";')
                        cursor.execute(f'CREATE TABLE "{project}_face_nodes" (id SERIAL PRIMARY KEY, original_node_id INTEGER, entity_idx INTEGER, frame_idx INTEGER, face_idx INTEGER, content JSONB);')
                        cursor.execute(f'CREATE TABLE "{project}_face_edges" (from_id INTEGER, to_id INTEGER);')
                        connection.commit()
                    except Exception as e:
                        print(f"Error recreating face tables: {e}")
                        connection.rollback()

                    face_ids = []
                    for f in faces:
                        node_id, entity_idx, is_frame, frame_idx, face_idx, embedding, face = f
                        # Fetch parent content to inherit geolocation
                        select_query = f'SELECT content FROM "{project}_nodes" WHERE id = %s;'
                        cursor.execute(select_query, (node_id,))
                        parent_content_row = cursor.fetchone()
                        if parent_content_row:
                            parent_content = parent_content_row[0]
                            geolocation = None
                            if is_frame:
                                entities_parent = parent_content
                                if 'profileData' in parent_content:
                                    for key in parent_content['profileData']:
                                        if 'entities' in parent_content['profileData'][key]:
                                            entities_parent = parent_content['profileData'][key]
                                            break
                                for ent in entities_parent.get('entities', []):
                                    if ent.get('index') == entity_idx:
                                        if 'frame_geolocation' in ent and frame_idx in ent['frame_geolocation']:
                                            geolocation = ent['frame_geolocation'][frame_idx]
                                        break
                            else:
                                if entity_idx == 1 and 'geolocation' in parent_content:
                                    geolocation = parent_content['geolocation']
                                else:
                                    entities_parent = parent_content
                                    if 'profileData' in parent_content:
                                        for key in parent_content['profileData']:
                                            if 'entities' in parent_content['profileData'][key]:
                                                entities_parent = parent_content['profileData'][key]
                                                break
                                    for ent in entities_parent.get('entities', []):
                                        if ent.get('index') == entity_idx:
                                            if 'geolocation' in ent:
                                                geolocation = ent['geolocation']
                                            break
                            if geolocation:
                                face['geolocation'] = geolocation
                        content_json = json.dumps(convert_to_serializable(face))
                        insert_query = f'INSERT INTO "{project}_face_nodes" (original_node_id, entity_idx, frame_idx, face_idx, content) VALUES (%s, %s, %s, %s, %s) RETURNING id;'
                        cursor.execute(insert_query, (node_id, entity_idx, frame_idx, face_idx, content_json))
                        face_id = cursor.fetchone()[0]
                        face_ids.append(face_id)
                    connection.commit()

                    # Insert into face_edges based on cosine similarity threshold
                    threshold = 0.6  # Cosine distance threshold for considering faces similar (adjust as needed)
                    edges = []
                    for i in range(len(faces)):
                        for j in range(i + 1, len(faces)):
                            dist = dist_matrix[i, j]
                            if dist < threshold:
                                edges.append((face_ids[i], face_ids[j]))
                                edges.append((face_ids[j], face_ids[i]))  # Bidirectional for DiGraph compatibility
                    if edges:
                        insert_edge_query = f'INSERT INTO "{project}_face_edges" (from_id, to_id) VALUES (%s, %s);'
                        cursor.executemany(insert_edge_query, edges)
                        connection.commit()
                    print("Face nodes and edges populated in database.")

                    # === Added: Prepare and send graph data for frontend ===
                    nodes = []
                    edges = []
                    for i, (node_id, entity_idx, is_frame, frame_idx, face_idx, _, _) in enumerate(faces):
                        face_id = f"node:{node_id}|entity:{entity_idx}|is_frame:{is_frame}|frame:{frame_idx}|face:{face_idx}"
                        label = f"N{node_id} E{entity_idx} F{face_idx}" + (f" Fr{frame_idx}" if is_frame else "")
                        nodes.append({"id": face_id, "label": label, "group": "face"})
                    for i in range(len(faces)):
                        for j in range(i + 1, len(faces)):
                            dist = dist_matrix[i, j]
                            if dist < cluster_threshold:
                                face1_node, face1_entity, face1_is_frame, face1_frame, face1_face = faces[i][:5]
                                face1 = f"node:{face1_node}|entity:{face1_entity}|is_frame:{face1_is_frame}|frame:{face1_frame}|face:{face1_face}"
                                face2_node, face2_entity, face2_is_frame, face2_frame, face2_face = faces[j][:5]
                                face2 = f"node:{face2_node}|entity:{face2_entity}|is_frame:{face2_is_frame}|frame:{face2_frame}|face:{face2_face}"
                                edges.append({"from": face1, "to": face2, "weight": 1 - dist})
                    graph_data = {"type": "graph_data", "nodes": nodes, "edges": edges}
                    try:
                        response = requests.post('http://127.0.0.1:4000/graphdata', json=graph_data)
                        if response.status_code == 200:
                            print("Graph data sent successfully")
                        else:
                            print(f"Error sending graph data: {response.status_code}")
                    except Exception as e:
                        print(f"Error sending graph data: {e}")

                    graphs_dir = os.path.join(project_dir, 'graphs')
                    os.makedirs(graphs_dir, exist_ok=True)
                    visualization_paths = []
                    fig = plt.figure(figsize=(12, 8))
                    dn = sch.dendrogram(linkage)
                    plt.title('Dendrogram of Face Clustering')
                    dendro_path = os.path.join(graphs_dir, 'face_dendrogram.png')
                    plt.savefig(dendro_path)
                    plt.close()
                    visualization_paths.append(dendro_path)
                    cluster_counts = Counter(clusters)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(range(1, len(cluster_counts)+1), list(cluster_counts.values()), tick_label=list(cluster_counts.keys()))
                    ax.set_xlabel('Cluster ID')
                    ax.set_ylabel('Number of Faces')
                    ax.set_title('Face Cluster Sizes')
                    cluster_size_path = os.path.join(graphs_dir, 'face_cluster_sizes.png')
                    plt.savefig(cluster_size_path)
                    plt.close()
                    visualization_paths.append(cluster_size_path)
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.pie(cluster_counts.values(), labels=cluster_counts.keys(), autopct='%1.1f%%', startangle=90)
                    ax.set_title('Face Cluster Distribution')
                    cluster_pie_path = os.path.join(graphs_dir, 'face_cluster_pie.png')
                    plt.savefig(cluster_pie_path)
                    plt.close()
                    visualization_paths.append(cluster_pie_path)
                    if len(embeddings) > 1:
                        pca = PCA(n_components=2)
                        embed_2d = pca.fit_transform(embeddings)
                        fig, ax = plt.subplots(figsize=(10, 8))
                        scatter = ax.scatter(embed_2d[:,0], embed_2d[:,1], c=clusters, cmap='viridis')
                        plt.colorbar(scatter)
                        ax.set_title('PCA Projection of Face Embeddings')
                        pca_path = os.path.join(graphs_dir, 'face_pca.png')
                        plt.savefig(pca_path)
                        plt.close()
                        visualization_paths.append(pca_path)
                    if len(embeddings) > 2:
                        tsne = TSNE(n_components=2, random_state=42)
                        embed_tsne = tsne.fit_transform(embeddings)
                        fig, ax = plt.subplots(figsize=(10, 8))
                        scatter = ax.scatter(embed_tsne[:,0], embed_tsne[:,1], c=clusters, cmap='viridis')
                        plt.colorbar(scatter)
                        ax.set_title('t-SNE Projection of Face Embeddings')
                        tsne_path = os.path.join(graphs_dir, 'face_tsne.png')
                        plt.savefig(tsne_path)
                        plt.close()
                        visualization_paths.append(tsne_path)
                    print("\n=== Sending Face Clustering visualizations to endpoint ===")
                    images = []  # List to collect base64 strings
                    for viz_path in visualization_paths:
                        if os.path.exists(viz_path):
                            with open(viz_path, 'rb') as f:
                                img_data = f.read()
                                base64_str = base64.b64encode(img_data).decode('utf-8')
                                images.append(base64_str)  # Add to images list
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
                    print(f"\nGenerated {len(visualization_paths)} face recognition visualizations")
                except Exception as cluster_e:
                    print(f"Error during clustering: {cluster_e}")

        # Face network analysis inside face clustering
        print(f"\nPerforming face network analysis for project: {project}")
        try:
            # Query face nodes and edges
            cursor.execute(f'SELECT id, original_node_id, content FROM "{project}_face_nodes";')
            nodes = cursor.fetchall()
            cursor.execute(f'SELECT from_id, to_id FROM "{project}_face_edges";')
            edges_list = cursor.fetchall()

            # Check for empty graph early
            if not nodes or not edges_list:
                print("No nodes or edges found; skipping centrality computations and plotting.")
                # Optionally update DB with default centralities (0) for all nodes
                cursor.execute(f'SELECT id FROM "{project}_nodes";')
                for row in cursor.fetchall():
                    original_node_id = row[0]
                    cursor.execute(f'SELECT content FROM "{project}_nodes" WHERE id = %s;', (original_node_id,))
                    content = cursor.fetchone()[0]
                    content['centralities'] = {
                        'face_betweenness': 0,
                        'face_closeness': 0,
                        'face_in_degree': 0,
                        'face_out_degree': 0,
                        'face_eigenvector': 0,
                        'face_cluster': 0,
                        'face_katz': 0,
                    }
                    cursor.execute(f'UPDATE "{project}_nodes" SET content = %s WHERE id = %s;', (json.dumps(content), original_node_id))
                connection.commit()
                print("Set default centralities for all nodes due to empty graph.")
                return

            # Create graph
            G = nx.DiGraph()
            for node_id, original_node_id, content in nodes:
                # Debug: Check if content contains image data that might cause issues
                if 'image' in content or 'img' in content:
                    print(f"Warning: Image data found in node {node_id} content: {list(content.keys())}")
                G.add_node(node_id, original_node_id=original_node_id, content=content)
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
            louvain = nx.community.louvain_communities(G.to_undirected())  # Convert to undirected for community detection
            print(f"Number of Louvain communities: {len(louvain)}")

            if not nx.is_strongly_connected(G):
                print("Graph is not strongly connected. Computing eigenvector centrality per component.")
                eigen = {}
                for component in nx.strongly_connected_components(G):
                    subgraph = G.subgraph(component)
                    if len(subgraph) > 1:
                        try:
                            comp_eigen = nx.eigenvector_centrality(subgraph, max_iter=50000, tol=1e-03)
                        except nx.PowerIterationFailedConvergence as e:
                            print(f"Power iteration failed for component {component}: {e}. Setting to 0.")
                            comp_eigen = {node: 0.0 for node in component}
                        eigen.update(comp_eigen)
                    else:
                        eigen.update({node: 0.0 for node in component})
            else:
                try:
                    eigen = nx.eigenvector_centrality(G, max_iter=50000, tol=1e-03)
                except nx.PowerIterationFailedConvergence as e:
                    print(f"Power iteration failed for full graph: {e}. Setting all to 0.")
                    eigen = {node: 0.0 for node in G.nodes}

            # Update DB
            centrality_map = defaultdict(list)
            for node_id in G.nodes:
                cursor.execute(f'SELECT original_node_id FROM "{project}_face_nodes" WHERE id = %s;', (node_id,))
                row = cursor.fetchone()
                if not row:
                    print(f"No original_node_id found for face node {node_id}")
                    continue
                original_node_id = row[0]
                cent = {
                    'face_betweenness': btwn.get(node_id, 0),
                    'face_closeness': close.get(node_id, 0),
                    'face_in_degree': in_deg.get(node_id, 0),
                    'face_out_degree': out_deg.get(node_id, 0),
                    'face_eigenvector': eigen.get(node_id, 0),
                    'face_cluster': clust.get(node_id, 0),
                    'face_katz': katz.get(node_id, 0),
                }
                centrality_map[original_node_id].append(cent)

            for original_node_id, cent_list in centrality_map.items():
                try:
                    aggregated_cent = {}
                    if cent_list:
                        for key in cent_list[0].keys():
                            values = [cent[key] for cent in cent_list]
                            aggregated_cent[key] = sum(values) / len(values)
                    else:
                        aggregated_cent = {k: 0 for k in ['face_betweenness', 'face_closeness', 'face_in_degree', 'face_out_degree', 'face_eigenvector', 'face_cluster', 'face_katz']}
                    cursor.execute(f'SELECT content FROM "{project}_nodes" WHERE id = %s;', (original_node_id,))
                    row = cursor.fetchone()
                    if row:
                        content = row[0]
                        content['centralities'] = aggregated_cent
                        cursor.execute(f'UPDATE "{project}_nodes" SET content = %s WHERE id = %s;', (json.dumps(content), original_node_id))
                        connection.commit()
                    else:
                        print(f"No node found in {project}_nodes for original_node_id {original_node_id}")
                except Exception as e:
                    print(f"Error updating face centralities for node {original_node_id}: {e}")

            print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges")

            # Generate histograms of centralities
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
                hist_path = os.path.join(graphs_dir, 'face_centralities_hist.png')
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
            for node_id, original_node_id, content in nodes:
                lat = lon = None
                if 'geolocation' in content:
                    geo = content['geolocation']
                    lat = geo.get('lat')
                    lon = geo.get('lon')
                if lat is not None and lon is not None:
                    geo_data.append({
                        '_id': node_id,
                        'lat': lat,
                        'lon': lon,
                        'lang': None,
                        'geocoded_location': None,
                        'flg': 0
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
                        edge_color = plt.get_cmap("Paired")(9)
                        node_color = plt.get_cmap("Paired")(6)
                        try:
                            countries = gpd.read_file("./ne_10m_admin_0_countries/")
                            fig, ax = plt.subplots(figsize=(16, 9))
                            fig.set_facecolor(c_background)
                            countries.plot(ax=ax, color=c_countries, edgecolor=c_borders, linewidth=0.1)
                            ax.plot(df_coordinates.x, df_coordinates.y, marker="o", markersize=1, color=node_color, linestyle="None", alpha=0.45)
                            ax.plot(df_bundled.x, df_bundled.y, color=edge_color, alpha=0.85, linewidth=0.1)
                            ax.axis("off")
                            handles = [
                                mpl_patches.Patch(facecolor=edge_color, label="Face Similarity"),
                                mpl_patches.Patch(facecolor=node_color, label="Faces in Network"),
                            ]
                            ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.2, 0.2))
                            path = os.path.join(graphs_dir, 'face_nodesundedges.png')
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
                    print(f"Face graph images sent successfully ({len(images)} images)")
                else:
                    print(f"Error sending face graph images: {response.status_code}")
            else:
                print("No images generated for sending.")
            print("Face network analysis completed.")
        except Exception as e:
            print(f"Error during face network analysis: {e}")

    connection.close()

if __name__ == "__main__":
    main()
