'''Project globals'''

# Paths
WORKING_DIRECTORY='..'
DATA_DIRECTORY=f'{WORKING_DIRECTORY}/data'
RAW_DATA_DIRECTORY=f'{DATA_DIRECTORY}/raw'
INTERIM_DATA_DIRECTORY=f'{DATA_DIRECTORY}/interim'
PROCESSED_DATA_DIRECTORY=f'{DATA_DIRECTORY}/processed'
MODEL_DIRECTORY=f'{WORKING_DIRECTORY}/models'

# Data files
RAW_INCIDENTS_MDB_FILE=f'{RAW_DATA_DIRECTORY}/avall.mdb'
RAW_INCIDENTS_CSV_FILE=f'{RAW_DATA_DIRECTORY}/incidents.csv'
EXTRACTED_INCIDENTS_FILE=f'{INTERIM_DATA_DIRECTORY}/incidents.csv'
RAW_ONTIME_CSV_FILE=f'{RAW_DATA_DIRECTORY}/ontime.csv'
EXTRACTED_ONTIME_FILE=f'{INTERIM_DATA_DIRECTORY}/ontime.csv'
COMBINED_DATAFILE=f'{PROCESSED_DATA_DIRECTORY}/combined_data.csv'
ENCODED_DATAFILE=f'{PROCESSED_DATA_DIRECTORY}/all_encoded.csv'
TRAINING_DATAFILE=f'{PROCESSED_DATA_DIRECTORY}/train_encoded.csv'
TESTING_DATAFILE=f'{PROCESSED_DATA_DIRECTORY}/test_encoded.csv'
MODEL=f'{MODEL_DIRECTORY}/model.pkl'

# Resource URLs
INCIDENT_DATA_URL='https://data.ntsb.gov/avdata/FileDirectory/DownloadFile?fileID=C%3A%5Cavdata%5Cavall.zip'
ONTIME_DATA_URL='https://www.bts.gov/browse-statistical-products-and-data/bts-publications/airline-service-quality-performance-234-time'
ONTIME_DATA_LINK_PREFIX='https://www.bts.dot.gov'

# Number of on-time performance files to download and parse
ONTIME_FILES=3