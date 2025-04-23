'''Functions to download extract and parse data.'''

import io
import glob
import zipfile
import os

import requests
from bs4 import BeautifulSoup, SoupStrainer
import pandas as pd
from access_parser import AccessParser
from urllib.parse import urljoin


def download_data(url:str) -> None:
    '''Downloads and extract zipfile from url.'''
    
    # Get the archive from URL
    response=requests.get(url, timeout=10)

    # Extract to disk
    archive=zipfile.ZipFile(io.BytesIO(response.content))
    archive.extractall('../data/raw/')


def parse_mdb(file:str) -> None:
    '''Parses MDB file and saves aircraft table to csv.'''

    # Load database and extract aircraft table
    db=AccessParser(f'../data/raw/{file}')
    table=db.parse_table('aircraft')

    # Convert to dataframe and save as csv
    table_df=pd.DataFrame.from_dict(table)
    table_df.to_csv('../data/raw/aircraft.csv', index=False)


def get_ontime_links(url:str) -> list:
    '''Uses requests and beautifulsoup to parse download links for on-time
    performance from bts.gov site'''

    response=requests.get(url)
    html_content=response.text
    soup=BeautifulSoup(html_content, 'html.parser', parse_only=SoupStrainer('a'))
    
    links=[]

    for link in soup:
        if link.has_attr('href'):
            link_text=link['href']
            if link_text.split('.')[-1] == 'zip':
                links.append(link_text)

    return links


def download_ontime_data(links:list) -> None:
    '''Takes list of data download urls, download the files to disk.'''

    for link in links:
        complete_link=f'https://www.bts.dot.gov/{link}'
        response=requests.get(complete_link, timeout=10)
        archive=zipfile.ZipFile(io.BytesIO(response.content))
        archive.extractall('../data/raw/')


def read_asc_datafiles(n_files:int) -> pd.DataFrame:
    '''Reads .asc files from raw data directory, combines into
    pandas dataframe.'''

    data_dfs=[]
    asc_files=glob.glob('../data/raw/*.asc')

    for asc_file in asc_files[:n_files]:
        print(asc_file)

        data_df=pd.read_table(asc_file, sep='|', low_memory=False)
        data_dfs.append(data_df)

    data_df=pd.concat(data_dfs, axis=0)

    return data_df

def download_airplane_regis(download_url: str, extract_to: str = "../data/raw/") -> None:
    
        complete_link = download_url
        response=requests.get(complete_link, timeout=10)
        archive=zipfile.ZipFile(io.BytesIO(response.content))
        archive.extractall(extract_to)


def unzip_files(zip_folder: str, extract_to: str, separate_folders: bool = True, delete_zip: bool = True):
    """
    Unzips all .zip files in the specified folder, skipping any bad zip files.

    Parameters:
    - zip_folder (str): Path to the folder containing zip files.
    - extract_to (str): Path to the folder where contents should be extracted.
    - separate_folders (bool): If True, creates a subfolder for each zip file. 
                               If False, extracts all into the same folder.
    - delete_zip (bool): If True, deletes the zip file after extraction.
    """
    os.makedirs(extract_to, exist_ok=True)

    for filename in os.listdir(zip_folder):
        if filename.endswith('.zip'):
            zip_path = os.path.join(zip_folder, filename)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    if separate_folders:
                        folder_name = os.path.splitext(filename)[0]
                        output_folder = os.path.join(extract_to, folder_name)
                        os.makedirs(output_folder, exist_ok=True)
                        zip_ref.extractall(output_folder)
                    else:
                        zip_ref.extractall(extract_to)

                print(f"Extracted: {filename}")

                if delete_zip:
                    os.remove(zip_path)
                    print(f"Deleted: {filename}")

            except zipfile.BadZipFile:
                print(f"Skipped (Bad Zip): {filename}")
            except Exception as e:
                print(f"Skipped ({filename}) due to error: {e}")


def combine_csvs_from_subfolders_chunked(parent_folder, output_filename="combined_data.csv", chunk_size=10000):
    """
    Combines all CSV files from subfolders using chunking and writes to a single CSV file.
    
    Args:
        parent_folder (str): Path to the parent folder containing subfolders with CSVs.
        output_filename (str): Name of the final combined CSV file.
        chunk_size (int): Number of rows per chunk to process.
    """
    output_path = os.path.join(parent_folder, output_filename)
    first_write = True

    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".csv"):
                    csv_path = os.path.join(folder_path, file_name)
                    print(f"Processing {csv_path}...")

                    try:
                        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
                            chunk.to_csv(output_path, mode='a', index=False, header=first_write)
                            first_write = False
                    except Exception as e:
                        print(f"Failed to process {csv_path}: {e}")

    print(f"\n✅ Combined CSV saved to: {output_path}")