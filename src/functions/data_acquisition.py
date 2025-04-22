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

    # Get download links page as HTML string
    response=requests.get(url)
    html_content=response.text

    # Convert to BeautifulSoup object and get all links from page by taking only <a> tags.
    soup=BeautifulSoup(html_content, 'html.parser', parse_only=SoupStrainer('a'))
    
    # Loop on the links and collect those that point to a zip file
    links=[]

    for link in soup:
        if link.has_attr('href'):
            link_text=link['href']
            if link_text.split('.')[-1] == 'zip':
                links.append(link_text)

    return links


def download_ontime_data(links:list) -> None:
    '''Takes list of data download urls, download the files to disk.'''

    # Loop on list of links
    for link in links:

        # Download the zip file
        complete_link=f'https://www.bts.dot.gov/{link}'
        response=requests.get(complete_link, timeout=10)

        # Extract the zip file to the raw data directory
        archive=zipfile.ZipFile(io.BytesIO(response.content))
        archive.extractall('../data/raw/')


def read_asc_datafiles(n_files:int) -> pd.DataFrame:
    '''Reads .asc files from raw data directory, combines into
    pandas dataframe.'''

    # Get list of ASCII files from raw data directory
    data_dfs=[]
    asc_files=glob.glob('../data/raw/*.asc')

    # Loop on the ASCII data files
    for asc_file in asc_files[:n_files]:
        print(asc_file)

        # Read the file into a Pandas dataframe and collect in list
        data_df=pd.read_table(asc_file, sep='|', low_memory=False)
        data_dfs.append(data_df)

    # Combine the list of Pandas dataframes and clean the index
    data_df=pd.concat(data_dfs, axis=0)
    data_df.reset_index(inplace=True, drop=True)

    return data_df

