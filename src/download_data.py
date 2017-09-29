import os
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
from time import time

start = datetime.datetime(2005, 1, 1)
end = datetime.datetime(2017, 6, 1)

export_file_name = 'data/snp500_raw.npz'
cont_export = True

excluded = []
equity_symbols = ['^GSPC']
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

# names we want to show in the dataset
col_names = ['open', 'high','low', 'close', 'adjclose', 'volume']
# extracted names
cols_to_extract = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

def get_data(equity_symbols, excluded, count):
    first_run = os.path.exists(export_file_name) == False
    if first_run:
        cur_sym = '^GSPC'
        syms = []
    else:
        with np.load(export_file_name) as file:
            timestamps = file['timestamps']
            syms       = list(file['syms'])
            data       = file['data']
        cur_sym = [s for s in equity_symbols if s not in syms]
        if len(cur_sym) == 0:
            return True, equity_symbols, excluded, 0
        cur_sym = cur_sym[0]

    if count > 10:
        print('Failed to get', cur_sym)
        excluded.append(cur_sym)
        equity_symbols = [sym for sym in equity_symbols if sym not in excluded]
        return False, equity_symbols, excluded, 0

    print('Getting', cur_sym, round(time() - starttime, 0), 'seconds')
    data_new = web.DataReader(cur_sym, 'yahoo', start, end)

    # ^GSPC is tested and should not have null for any time frames
    # it is the goddamnned snp500 index
    if np.sum(np.sum(pd.isnull(data_new))) != 0:
        excluded.append(cur_sym)
        equity_symbols = [sym for sym in equity_symbols if sym not in excluded]
        print('Nan detected in', cur_sym)
        return False, equity_symbols, excluded, 0

    if first_run:
        timestamps = data_new.index
    data_new = data_new[cols_to_extract]

    # here we attempt to account for stock splits and joins
    for i in range(len(data_new) - 1):
        t0_close = data_new.loc[timestamps[i], 'Adj Close']
        t1_open = data_new.loc[timestamps[i + 1], 'Open']

        ratio = t0_close / t1_open

        is_split = ratio > 1.9
        is_join = 1 / 1.9 > ratio

        if is_split | is_join:
            data_new.loc[i + 1:, :] *= ratio

    syms.append(cur_sym)
    data_new = np.expand_dims(data_new, 1)

    if first_run:
        data = data_new
    else:
        data = np.concatenate((data, data_new), axis = 1)

    np.savez(export_file_name, timestamps=timestamps, syms=syms, col_names=col_names, data=data)

    return False, equity_symbols, excluded, 0

starttime = time()
done = False
count = 0

while not done:
    try:
        done, equity_symbols, excluded, count = get_data(equity_symbols, excluded, count)
    except:
        count += 1
        print('Try again')

print('All done', round(time() - starttime, 0), 'seconds')
print('Excluded', excluded)
