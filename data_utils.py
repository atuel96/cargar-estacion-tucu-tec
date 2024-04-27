import numpy as np
import pandas as pd
from pathlib import Path

def read_year_files(year : int, datafolder : str | Path):
    year_str = str(year)[-2:]
    datafolder = Path(datafolder)
    datafiles = [file for file in datafolder.iterdir() if file.is_file() and file.suffix == f".{year_str}A"]
    datafiles.sort()

    return datafiles
    #for file in datafiles:
    #    print(file)

def load_and_process_file(filepath : str | Path):
    df = pd.read_csv(filepath, sep=" ", header=None, usecols=[0, 1, 7], names=["seconds", "alpha", "TEC"])
    df_vTEC = df.query("alpha == 'Z00'")

    # Look for negative TEC values
    negative_TEC = df_vTEC["TEC"][df_vTEC["TEC"] < 0] 
    negative_values = negative_TEC.count()
    if negative_values:
        df_vTEC["TEC"][df_vTEC["TEC"] < 0] = None
        print(f"{negative_values} TEC negative values were replaced by NaNs for file {filepath}.") # Info
    
    # Fill non-existing values with Nan
    # Define the complete range of seconds (0 to 86400 with 5760 steps)
    total_seconds = 86400
    num_steps = 5760
    step_size = round(total_seconds/num_steps)
    complete_seconds = pd.Series(np.arange(0, total_seconds + 1, step_size))

    # Create a DataFrame with the complete range of seconds
    complete_df = pd.DataFrame({'seconds': complete_seconds})
    merged_df = pd.merge(complete_df, df_vTEC, on='seconds', how='left')
    merged_df.sort_values('seconds', inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    # Delete duplicates
    #n_duplicated_values = merged_df.duplicated().sum()
    return merged_df[["seconds", "TEC"]].drop_duplicates()




if __name__ == "__main__":
    #load_year_files(2017, "tucu")
    ...