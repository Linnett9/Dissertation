"""
Script to download the metadata for specific 
TCIA datasets. The metadata collected 
at this stage is the Modality, SeriesInstanceUID,  ImageCount, 
BodyPartExamined, SeriesDescription, PatientID, and TotalSizeInBytes.
"""
# Code adapted from the get_collection_metadata script
# Source: https://github.com/ExampleUser/tcia-download-tools/blob/main/get_collection_metadata.py
# Author: cdmacfadyen

import aiohttp
import asyncio
from tciaclient import TCIAClient
import pandas as pd
import numpy as np
from io import StringIO
import json
import logging
import os

# Configure logging
logging.basicConfig(filename='metadata_download.log', filemode="w", level=logging.DEBUG, format='%(levelname)s:%(message)s')

async def fetch_series_size(session, tcia_client, uid):
    url = f"{tcia_client.baseUrl}/query/getSeriesSize?SeriesInstanceUID={uid}"
    async with session.get(url) as response:
        size_dict = await response.json()
        size = int(float(size_dict[0]["TotalSizeInBytes"]))
        return uid, size

async def get_series_sizes(metadata, tcia_client):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for uid in metadata["SeriesInstanceUID"]:
            task = asyncio.ensure_future(fetch_series_size(session, tcia_client, uid))
            tasks.append(task)
        size_list = await asyncio.gather(*tasks)
        size_dict = {uid: size for uid, size in size_list}
        return size_dict

tcia_client = TCIAClient(baseUrl="https://services.cancerimagingarchive.net/services/v4", resource="TCIA")

specific_datasets = ["CMMD"]

for collection in specific_datasets:
    try:
        print(f"Processing collection: {collection}")
        logging.info(f"Processing collection: {collection}")
        series_response = tcia_client.get_series(collection=collection, outputFormat="json")
        response_as_string = series_response.read().decode()
        response_df = pd.read_json(StringIO(response_as_string))

        columns_to_take = ["Modality", "SeriesInstanceUID", "ImageCount", "BodyPartExamined", "SeriesDescription", "PatientID"]
        metadata = pd.DataFrame()
        column_length = len(response_df["Modality"])
        for column in columns_to_take:
            try:
                metadata[column] = response_df[column]
            except KeyError:
                metadata[column] = [np.NaN] * column_length  # makes a list `column_length` long of NaN

        size_dict = asyncio.run(get_series_sizes(metadata, tcia_client))
        metadata["TotalSizeInBytes"] = metadata["SeriesInstanceUID"].map(size_dict)
        os.makedirs("./metadata", exist_ok=True)
        metadata.to_csv(f"./metadata/{collection}.csv")
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(e)
        logging.error(f"Error processing collection {collection}: {e}")
