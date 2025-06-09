import argparse
import math
import time
from urllib.parse import quote

import pandas as pd
import requests
from tqdm import tqdm

from code_base.utils import load_json

SLEEP_TIME = 2  # seconds
LONG_SLEEP_TIME = 30  # seconds


def handle_request(url):
    success = True
    try:
        time.sleep(SLEEP_TIME)
        response = requests.get(url)
        if response.status_code == 429:
            print(f"Rate limit exceeded. Waiting for {LONG_SLEEP_TIME} seconds...")
            time.sleep(LONG_SLEEP_TIME)
            response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to make request. Status code: {response.status_code}")
            success = False
    except Exception as e:
        print(f"An error occurred: {e}")
        success = False
        response = None
    return success, response


def search_inaturalist_audio(common_name, downloaded_files=None, accepted_licenses=None, max_pages=None):
    base_url = "https://api.inaturalist.org/v1/observations"
    per_page = 30
    results_list = []

    url = f"{base_url}?taxon_name={quote(common_name)}&media_type=sound&per_page={per_page}&page=1"
    time.sleep(SLEEP_TIME)
    success, response = handle_request(url)
    if not success:
        print(f"Failed to fetch data for {common_name} on page 1")
        return pd.DataFrame()

    data = response.json()
    total_results = data.get("total_results", 0)
    if total_results == 0:
        return pd.DataFrame()

    if max_pages is None:
        max_pages = math.ceil(total_results / per_page)

    for page in tqdm(range(1, max_pages + 1)):
        url = f"{base_url}?taxon_name={quote(common_name)}&media_type=sound&per_page={per_page}&page={page}"
        success, response = handle_request(url)
        if not success:
            print(f"Failed to fetch data for {common_name} on page {page}")
            continue

        page_data = response.json()
        observations = page_data.get("results", [])
        if not observations:
            continue

        for obs in observations:
            sounds = obs.get("sounds", [])
            if not sounds:
                continue  # skip observations with no sounds

            for sound in sounds:
                license_code = sound.get("license_code", "")
                if accepted_licenses and license_code not in accepted_licenses:
                    continue

                audio_url = sound.get("file_url")
                filename = audio_url.split("/")[-1] if audio_url else None
                if not audio_url or (downloaded_files and filename in downloaded_files):
                    continue

                coordinates = obs.get("geojson", {})
                if coordinates is not None:
                    coordinates = coordinates.get("coordinates")
                else:
                    coordinates = [None, None]
                longitude, latitude = coordinates[0], coordinates[1]

                results_list.append(
                    {
                        "primary_label": common_name,
                        "secondary_labels": None,
                        "type": "audio",
                        "filename": filename,
                        "collection": "iNat",
                        "rating": obs.get("quality_grade", None),
                        "url": audio_url,
                        "latitude": latitude,
                        "longitude": longitude,
                        "scientific_name": obs.get("taxon", {}).get("name"),
                        "common_name": obs.get("taxon", {}).get("preferred_common_name", common_name),
                        "author": obs.get("user", {}).get("login"),
                        "license": license_code,
                    }
                )

    return pd.DataFrame(results_list)


def main():
    parser = argparse.ArgumentParser(description="Download audio data from iNaturalist.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to JSON file with list of common names.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the resulting CSV file.")
    args = parser.parse_args()

    # Load common names from the input JSON file
    common_names = load_json(args.input_json)

    if not isinstance(common_names, list):
        raise ValueError("The input JSON file must contain a list of common names.")

    # Collect data for all common names
    all_results = []
    for common_name in common_names:
        print(f"Processing: {common_name}")
        results = search_inaturalist_audio(common_name)
        all_results.append(results)

    # Combine all results into a single DataFrame and save to CSV
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
