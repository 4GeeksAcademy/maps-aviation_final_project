"""Functions to download, extract, and parse aviation-related data."""

# Import necessary libraries for file handling, web scraping, and data processing
import io
import glob
import zipfile
from pathlib import Path

import requests
import pandas as pd
from bs4 import BeautifulSoup, SoupStrainer
from access_parser import AccessParser


def download_data(url: str, raw_data_directory: str, raw_incidents_mdb_file: str) -> None:
    """
    Downloads and extracts a zip file from the given URL, saving it to the raw data directory.

    Parameters:
    - url (str): The URL of the zip file to download.
    - raw_data_directory (str): The directory where extracted files will be saved.
    - raw_incidents_mdb_file (str): The expected MDB file path to check if download is necessary.
    """
    
    # Only download the file if it doesn't already exist
    if not Path(raw_incidents_mdb_file).is_file():
        # Get the archive from the URL
        response = requests.get(url, timeout=10)

        # Extract the zip file to disk
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        archive.extractall(raw_data_directory)


def parse_mdb(raw_incidents_mdb_file: str, raw_incidents_csv_file: str) -> None:
    """
    Parses an MDB file and saves the aircraft table as a CSV file.

    Parameters:
    - raw_incidents_mdb_file (str): Path to the MDB file containing incident data.
    - raw_incidents_csv_file (str): Path where the parsed CSV file will be saved.
    """
    
    # Only parse the file if the CSV file doesn't already exist
    if not Path(raw_incidents_csv_file).is_file():
        # Load the database and extract the aircraft table
        db = AccessParser(raw_incidents_mdb_file)
        table = db.parse_table('aircraft')

        # Convert to a DataFrame and save as CSV
        table_df = pd.DataFrame.from_dict(table)
        table_df.to_csv(raw_incidents_csv_file, index=False)


def get_ontime_links(url: str) -> list:
    """
    Parses download links for on-time performance data from the BTS.gov site.

    Parameters:
    - url (str): The URL of the page containing download links.

    Returns:
    - links (list): A list of URLs pointing to zip files.
    """
    
    # Get the download links page as an HTML string
    response = requests.get(url)
    html_content = response.text

    # Convert to a BeautifulSoup object and extract all <a> tags
    soup = BeautifulSoup(html_content, 'html.parser', parse_only=SoupStrainer('a'))
    
    # Collect links that point to zip files
    links = [link['href'] for link in soup if link.has_attr('href') and link['href'].endswith('.zip')]

    return links


def download_ontime_data(links: list, ontime_data_link_prefix: str, raw_data_directory: str) -> None:
    """
    Downloads zip files from a list of URLs and extracts them to the raw data directory.

    Parameters:
    - links (list): A list of URLs pointing to zip files.
    - ontime_data_link_prefix (str): The base URL prefix for constructing full download links.
    - raw_data_directory (str): The directory where extracted files will be saved.
    """
    
    # Loop through the list of links
    for link in links:
        file_path = Path(f"{raw_data_directory}{link.split('/')[-1]}")

        # Only download if the file doesn't already exist
        if not file_path.is_file():
            # Construct the full download link
            complete_link = f'{ontime_data_link_prefix}/{link}'
            response = requests.get(complete_link, timeout=10)

            # Extract the zip file to the raw data directory
            archive = zipfile.ZipFile(io.BytesIO(response.content))
            archive.extractall(raw_data_directory)


def parse_asc_datafiles(n_files: int, raw_data_directory: str, raw_ontime_csv_file: str) -> pd.DataFrame:
    """
    Reads .asc files from the raw data directory, combines them into a Pandas DataFrame, and saves as CSV.

    Parameters:
    - n_files (int): Number of files to process.
    - raw_data_directory (str): The directory containing .asc files.
    - raw_ontime_csv_file (str): Path where the combined CSV file will be saved.

    Returns:
    - data_df (pd.DataFrame): The combined DataFrame containing parsed data.
    """
    
    # Get a list of ASCII files from the raw data directory
    asc_files = glob.glob(f'{raw_data_directory}/*.asc')
    data_dfs = []

    # Loop through the ASCII data files (limited to n_files)
    for asc_file in asc_files[:n_files]:
        print(f"Processing file: {asc_file}")

        # Read the file into a Pandas DataFrame and collect in a list
        data_df = pd.read_table(asc_file, sep='|', low_memory=False)
        data_dfs.append(data_df)

    # Combine the list of DataFrames and reset the index
    data_df = pd.concat(data_dfs, axis=0)
    data_df.reset_index(inplace=True, drop=True)

    # Save the combined data as a CSV file
    data_df.to_csv(raw_ontime_csv_file, index=False)

    return data_df

