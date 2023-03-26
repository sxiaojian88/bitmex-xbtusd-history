import pandas as pd
from datetime import datetime, timezone, timedelta
from talib import abstract


# load dataframe from file
def load_df_from_csv_file(file):
    df = pd.read_csv(file, index_col=0, parse_dates=False)
    return df

# return df with only rows that are older than the current date
def filter_df_by_current_date(df,date):
    df = df[df['index'] < date.timestamp()]
    return df

ta_list = talib.get_functions()
for x in ta_list:
    try:
        output = eval('abstract.'+x+'(df)')
        output.name = x.lower() if type(output) == pd.core.series.Series else None
        df = pd.merge(df, pd.DataFrame(output), left_on = df.index, right_on = output.index)
        df = df.set_index('key_0')
    except:
        print(x)