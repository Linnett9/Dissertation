"""
Script that downloads all of the data from The Cancer Imaging 
Archive. Iterates over a csv containing 
the name of every dataset to download and requests the 
images from TCIA. Then unzips the images and places 
them in the correct directory for each modality. 

Optionally takes a command line argument to start the 
download at a specific dataset, instead of 
downloading all of them.
"""
# Code adapted from tcia_download_script_sync.py
# Source: https://github.com/cdmacfadyen/classify-modality/blob/main/src/tcia-download/tcia_download_script_sync.py
# Author: cdmacfadyen

"""
Script that downloads all of the data from The Cancer Imaging 
Archive. Iterates over a csv containing 
the name of every dataset to download and requests the 
images from TCIA. Then unzips the images and places 
them in the correct directory for each modality. 

Optionally takes a command line argument to start the 
download at a specific dataset, instead of 
downloading all of them.
"""
from tciaclient import TCIAClient
import urllib.request, urllib.error, urllib.parse, sys
import pandas as pd
import numpy as np
import time
import zipfile
from zipfile import ZipFile, BadZipFile
import os
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", required=False, default="", help="specific dataset to start from")
parser.add_argument("dataset", metavar="d", help="one of CBIS-DDSM or CMMD")
parser.add_argument("--out", required=True, help="directory to save data in")
args = parser.parse_args()

output_dir = args.out
dataset = args.dataset
logging.basicConfig(filename=f'{dataset}.log', filemode="w", level=logging.DEBUG, format='%(levelname)s:%(asctime)s %(message)s')

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

modalities = ["MR", "CT", "PT", "CR", "DX", "MG"]
limits = {key: 1e9 for key in modalities}

# Adjust limits for specific download requirements
limits["MR"] = 1e5
limits["CT"] = 5e4
limits["PT"] = 1e9
limits["MG"] = 1e9

# Read the CSV file
processed_collection_data = pd.read_csv(f"metadata/{dataset}.csv")

start_index = 0
if args.start:
    print(f"Start at {args.start}")
    series_uids = processed_collection_data["SeriesInstanceUID"]
    if args.start in series_uids.values:
        start_index = list(series_uids).index(args.start)
    else:
        print(f"Error: SeriesInstanceUID '{args.start}' not found in the dataset.")
        sys.exit(1)

tcia_client = TCIAClient(baseUrl="https://services.cancerimagingarchive.net/services/v4", resource="TCIA")
start = time.time()
bytes_downloaded = 0

total_images_downloaded = 0

for i in range(start_index, len(processed_collection_data)):
    modality = processed_collection_data["Modality"][i]
    series_uid = processed_collection_data["SeriesInstanceUID"][i]
    collection = dataset
    
    print(f"Downloading {collection} - {modality} - SeriesInstanceUID: {series_uid}")
    logging.info(f"Downloading {collection} - {modality} - SeriesInstanceUID: {series_uid}")
    
    zip_file_path = os.path.join(output_dir, f"temp-{collection}-{modality}-{i}.zip")
    tcia_client.get_image(series_uid, f"{output_dir}", zip_file_path)
    
    if not os.path.exists(zip_file_path):
        logging.error(f"Zip file not found: {zip_file_path}")
        continue

    if os.path.getsize(zip_file_path) == 0:
        logging.error(f"Zip file is empty: {zip_file_path}")
        os.remove(zip_file_path)
        continue

    try:
        with ZipFile(zip_file_path, "r") as zip_file:
            file_list = zip_file.namelist()
            print(f"Contents of zip file: {file_list}")
            logging.info(f"Contents of zip file: {file_list}")

            extract_path = os.path.join(output_dir, dataset)
            print(f"Extracting to: {extract_path}")
            logging.info(f"Extracting to: {extract_path}")
            
            os.makedirs(extract_path, exist_ok=True)
            zip_file.extractall(extract_path)
            
            # Check if files were actually extracted
            extracted_files = os.listdir(extract_path)
            print(f"Files in extraction directory: {extracted_files}")
            logging.info(f"Files in extraction directory: {extracted_files}")
            
            images_in_zip = len([f for f in extracted_files if f.lower().endswith(('.dcm', '.dicom'))])
            total_images_downloaded += images_in_zip
            
            logging.info(f"Extracted {images_in_zip} images from {collection} - {series_uid}")
        
        os.remove(zip_file_path)
    except BadZipFile:
        logging.error(f"Bad zip file: {collection} --- {series_uid}")
    except IOError as e:
        logging.error(f"IOError while processing {collection} --- {series_uid}: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error while processing {collection} --- {series_uid}: {str(e)}")

    bytes_downloaded += processed_collection_data['TotalSizeInBytes'][i]
    print(f"Downloaded {images_in_zip} images, Total: {total_images_downloaded}, {bytes_downloaded // 1000000}MB so far")

print(f"Total images downloaded: {total_images_downloaded}")
print(f"Downloaded {bytes_downloaded // 1000000000}GB in {time.time() - start} seconds")

# After the loop, check the total number of files in the output directory
total_files = sum([len(files) for r, d, files in os.walk(os.path.join(output_dir, dataset))])
print(f"Total files in output directory: {total_files}")