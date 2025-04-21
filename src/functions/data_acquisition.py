'''Functions to download extract and parse data.'''

import io
import zipfile

import requests
import pandas as pd
from access_parser import AccessParser


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
