import sys

sys.path.append('../Modules')
import pandas
from datafetch import DataFetch
from features import Features

d = DataFetch()
f = Features()
symbols_full = ['AAPL','AMD','NVDA','INTC','GOOGL']
symbols = ['AMD', 'NVDA', 'INTC']
start_date = '2000-02-01'
end_date = '2018-02-04'
moving_window = 20
y_target = 'close'


# Data fetch
# d.fetch_quandl_data(symbols, start_date, end_date, '../RawData')

# Add features
dfs = f.add_features_for_all_stocks(raw_data_directory='../RawData', symbols=symbols, moving_window=moving_window, output_directory='../RawData')

# for df in dfs:
#     df['y'] = df['close'].diff(periods=1)

# f.merge_df_by_time(df_list=dfs, output_directory='../RawData')

# Normalization
# f.normalize_for_all_stocks(raw_data_directory='../RawData', symbols=symbols, output_directory='../RawData')
# df = pandas.read_csv('../RawData/features.csv')
# print(df.head(5))

