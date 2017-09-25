import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
from time import time

start = datetime.datetime(2005, 1, 1)
end = datetime.datetime(2017, 9, 16)

export_file_name = 'data/SNP500_PRICE.csv'
cont_export = True

excluded = []
equity_symbols = [0, '^GSPC']
equity_symbols += [
    "MMM", "ABT", "ABBV", "ACN", "ATVI", "AYI", "ADBE", "AMD",
    "AAP", "AES", "AET", "AMG", "AFL", "A", "APD", "AKAM", "ALK",
    "ALB", "ARE", "ALXN", "ALGN", "ALLE", "AGN", "ADS", "LNT",
    "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AEE", "AAL", "AEP",
    "AXP", "AIG", "AMT", "AWK", "AMP", "ABC", "AME", "AMGN",
    "APH", "APC", "ADI", "ANDV", "ANSS", "ANTM", "AON", "AOS",
    "APA", "AIV", "AAPL", "AMAT", "ADM", "ARNC", "AJG", "AIZ",
    "T", "ADSK", "ADP", "AZO", "AVB", "AVY", "BHGE", "BLL", "BAC",
    "BK", "BCR", "BAX", "BBT", "BDX", "BRK-B", "BBY", "BIIB",
    "BLK", "HRB", "BA", "BWA", "BXP", "BSX", "BHF", "BMY", "AVGO",
    "BF-B", "CHRW", "CA", "COG", "CDNS", "CPB", "COF", "CAH",
    "CBOE", "KMX", "CCL", "CAT", "CBG", "CBS", "CELG", "CNC",
    "CNP", "CTL", "CERN", "CF", "SCHW", "CHTR", "CHK", "CVX",
    "CMG", "CB", "CHD", "CI", "XEC", "CINF", "CTAS", "CSCO", "C",
    "CFG", "CTXS", "CLX", "CME", "CMS", "COH", "KO", "CTSH", "CL",
    "CMCSA", "CMA", "CAG", "CXO", "COP", "ED", "STZ", "COO",
    "GLW", "COST", "COTY", "CCI", "CSRA", "CSX", "CMI", "CVS",
    "DHI", "DHR", "DRI", "DVA", "DE", "DLPH", "DAL", "XRAY",
    "DVN", "DLR", "DFS", "DISCA", "DISCK", "DISH", "DG", "DLTR",
    "D", "DOV", "DWDP", "DPS", "DTE", "DRE", "DUK", "DXC", "ETFC",
    "EMN", "ETN", "EBAY", "ECL", "EIX", "EW", "EA", "EMR", "ETR",
    "EVHC", "EOG", "EQT", "EFX", "EQIX", "EQR", "ESS", "EL", "ES",
    "RE", "EXC", "EXPE", "EXPD", "ESRX", "EXR", "XOM", "FFIV",
    "FB", "FAST", "FRT", "FDX", "FIS", "FITB", "FE", "FISV",
    "FLIR", "FLS", "FLR", "FMC", "FL", "F", "FTV", "FBHS", "BEN",
    "FCX", "GPS", "GRMN", "IT", "GD", "GE", "GGP", "GIS", "GM",
    "GPC", "GILD", "GPN", "GS", "GT", "GWW", "HAL", "HBI", "HOG",
    "HRS", "HIG", "HAS", "HCA", "HCP", "HP", "HSIC", "HSY", "HES",
    "HPE", "HLT", "HOLX", "HD", "HON", "HRL", "HST", "HPQ", "HUM",
    "HBAN", "IDXX", "INFO", "ITW", "ILMN", "IR", "INTC", "ICE",
    "IBM", "INCY", "IP", "IPG", "IFF", "INTU", "ISRG", "IVZ",
    "IRM", "JEC", "JBHT", "SJM", "JNJ", "JCI", "JPM", "JNPR",
    "KSU", "K", "KEY", "KMB", "KIM", "KMI", "KLAC", "KSS", "KHC",
    "KR", "LB", "LLL", "LH", "LRCX", "LEG", "LEN", "LVLT", "LUK",
    "LLY", "LNC", "LKQ", "LMT", "L", "LOW", "LYB", "MTB", "MAC",
    "M", "MRO", "MPC", "MAR", "MMC", "MLM", "MAS", "MA", "MAT",
    "MKC", "MCD", "MCK", "MDT", "MRK", "MET", "MTD", "MGM",
    "KORS", "MCHP", "MU", "MSFT", "MAA", "MHK", "TAP", "MDLZ",
    "MON", "MNST", "MCO", "MS", "MOS", "MSI", "MYL", "NDAQ",
    "NOV", "NAVI", "NTAP", "NFLX", "NWL", "NFX", "NEM", "NWSA",
    "NWS", "NEE", "NLSN", "NKE", "NI", "NBL", "JWN", "NSC",
    "NTRS", "NOC", "NRG", "NUE", "NVDA", "ORLY", "OXY", "OMC",
    "OKE", "ORCL", "PCAR", "PKG", "PH", "PDCO", "PAYX", "PYPL",
    "PNR", "PBCT", "PEP", "PKI", "PRGO", "PFE", "PCG", "PM",
    "PSX", "PNW", "PXD", "PNC", "RL", "PPG", "PPL", "PX", "PCLN",
    "PFG", "PG", "PGR", "PLD", "PRU", "PEG", "PSA", "PHM", "PVH",
    "QRVO", "PWR", "QCOM", "DGX", "Q", "RRC", "RJF", "RTN", "O",
    "RHT", "REG", "REGN", "RF", "RSG", "RMD", "RHI", "ROK",
    "COL", "ROP", "ROST", "RCL", "CRM", "SBAC", "SCG", "SLB",
    "SNI", "STX", "SEE", "SRE", "SHW", "SIG", "SPG", "SWKS",
    "SLG", "SNA", "SO", "LUV", "SPGI", "SWK", "SBUX", "STT",
    "SRCL", "SYK", "STI", "SYMC", "SYF", "SNPS", "SYY", "TROW",
    "TGT", "TEL", "FTI", "TXN", "TXT", "TMO", "TIF", "TWX", "TJX",
    "TMK", "TSS", "TSCO", "TDG", "TRV", "TRIP", "FOXA", "FOX",
    "TSN", "UDR", "ULTA", "USB", "UA", "UAA", "UNP", "UAL", "UNH",
    "UPS", "URI", "UTX", "UHS", "UNM", "VFC", "VLO", "VAR", "VTR",
    "VRSN", "VRSK", "VZ", "VRTX", "VIAB", "V", "VNO", "VMC", "WMT",
    "WBA", "DIS", "WM", "WAT", "WEC", "WFC", "HCN", "WDC", "WU",
    "WRK", "WY", "WHR", "WMB", "WLTW", "WYN", "WYNN", "XEL", "XRX",
    "XLNX", "XL", "XYL", "YUM", "ZBH", "ZION", "ZTS"
]
equity_symbols = [sym for sym in equity_symbols if sym not in excluded]

def get_data():
    df = pd.read_csv(export_file_name)
    sym_last = list(df['Sym']).pop()

    sym = equity_symbols[equity_symbols.index(sym_last) + 1]

    print(sym, round(time() - starttime, 0), 'seconds')

    data = web.DataReader(sym, 'yahoo', start, end)
    data['Sym'] = sym

    df = pd.concat([df, data])
    df.to_csv(export_file_name, index_label=False)

    return sym

def get_rand_number():
    method_index = np.random.rand(1)[0] * 3

    if method_index < 1:
        return np.random.rand(1)[0]
    elif method_index < 2:
        return np.random.randn(1)[0] + 1
    else:
        return np.random.ranf(1)[0]

starttime = time()

if cont_export:
    df = pd.read_csv(export_file_name)
else:
    df = pd.DataFrame(np.zeros((1,7)), index=[0], columns=['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Sym'])
    df.to_csv(export_file_name, index_label=False)

#web.get_data_yahoo('CAT', start, end).head()
sym_last = "0"

while sym_last != equity_symbols[len(equity_symbols) - 1]:
    try:
        sym_last = get_data()
    except:
        print('try again')

# config df to right format
# note that after this transformation you won't be able to continue exporting data unless you backup your data
df = pd.read_csv(export_file_name)
df = df[df['Sym'] != '0.0']
df['Timestamp'] = list(map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d'), df.index))
df = df[['Timestamp' ,'Sym' ,'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
df.index = range(len(df))

# do some checks on df

# save only after you have checked df
df.to_csv(export_file_name)
