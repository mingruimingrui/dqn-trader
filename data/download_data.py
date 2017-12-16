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

fees = pd.DataFrame(
    index=['bitstamp','btcn','btctrade','coinbase'],
    columns=['pair','BC_withdraw','BC_deposit','USD_withdraw','USD_deposit']
)

fees.loc['bitstamp'] = [0.0025,     0, 0,   0.02, 0.02]
fees.loc['btcn']     = [     0,     0, 0, 0.0038,    0]
fees.loc['btctrade'] = [0.002 , 0.001, 0,   0.01,    0]
fees.loc['coinbase'] = [  0.01,     0, 0,   0.01, 0.01]

fees['BC_deposit'] = 0.00001

fees.to_csv('bitcoin_fees.csv')
