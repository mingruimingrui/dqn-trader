"""
Go to https://api.bitcoincharts.com/v1/csv/ and download the following files

'bitstampUSD.csv.gz'
'btcnCNY.csv.gz'
'btctradeCNY.csv.gz'
'coinbaseUSD.csv.gz'

Convert them into npz files with the key 'data' and store them in data/raw/

There are no intensions of automating this script any time soon...
"""

DOWNLOAD_URL = 'https://api.bitcoincharts.com/v1/csv/'
FILES_TO_DOWNLOAD = [
    'bitstampUSD.csv.gz',
    'btcnCNY.csv.gz',
    'btctradeCNY.csv.gz',
    'coinbaseUSD.csv.gz'
]
