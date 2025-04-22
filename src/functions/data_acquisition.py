'''Functions to download extract and parse data.'''

import io
import glob
import zipfile

import requests
from bs4 import BeautifulSoup, SoupStrainer
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

