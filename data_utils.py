import numpy as np
import pandas as pd
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt

import logging
from datetime import datetime, timedelta
import gc



def get_tec_year_files(year: int, datafolder: str | Path):
    """
    Parameters
    ----------
    year : int
        Only the last two numbers are important.

    datafolder : str | Path
        The path of the folder where the year files are located.

    Returns
    -------
    List[Path] : A list with the Paths to the files.
    """
    year_str = str(year)[-2:]
    datafolder = Path(datafolder)
    datafiles = [
        file
        for file in datafolder.iterdir()
        if file.is_file() and file.suffix == f".{year_str}A"
    ]
    datafiles.sort()

    return datafiles


def read_and_process_tec_file(
    filepath: str | Path,
    load_with_polars=True,
):
    """
    Parameters
    ----------
    filepath : str | Path
        Path of the file.

    load_with_polars : bool
        if True then it uses polars library for reading files (~40% less time).

    Returns
    -------
    DataFrame : Pandas Dataframe with 'seconds' & 'TEC' columns.
    """
    if load_with_polars:
        df = pl.read_csv(
            filepath,
            separator=" ",
            has_header=False,
            columns=[0, 1, 7],
        ).to_pandas()
        df.columns = ["seconds", "alpha", "tec"]
    else:
        df = pd.read_csv(
            filepath,
            sep=" ",
            header=None,
            usecols=[0, 1, 7],
            names=["seconds", "alpha", "tec"],
        )
    df_vTEC = df.query("alpha == 'Z00'")  # take just vTEC

    # Delete duplicates
    n_duplicated_values = df_vTEC.duplicated().sum()
    if n_duplicated_values:
        logging.info(f"{n_duplicated_values} duplicated values inf file {filepath}.")

    df_vTEC = df_vTEC.loc[:, ["seconds", "tec"]].drop_duplicates()

    # Look for negative or zero TEC values
    negative_TEC = df_vTEC.loc[df_vTEC["tec"] <= 0, "tec"]
    negative_values = negative_TEC.count()
    if negative_values:
        df_vTEC.loc[df_vTEC["tec"] < 0, "tec"] = None
        logging.info(
            f"{negative_values} TEC negative values were replaced by NaNs for file {filepath}."
        )  # Info

    # frequency
    delta_t = int(df_vTEC.diff(1)["seconds"].mode()[0])

    # Fill non-existing values with Nan
    # Create a DataFrame with the complete range of seconds
    complete_df = create_seconds_df(delta_t=delta_t)
    merged_df = pd.merge(complete_df, df_vTEC, on="seconds", how="left")
    merged_df.sort_values("seconds", inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df[["seconds", "tec"]]


def create_seconds_df(delta_t: int) -> pd.DataFrame:
    """
    create a DataFrame with only sencods from 1 day
    """
    total_seconds_1_day = 86400  # 1 day in seconds
    num_steps = round(total_seconds_1_day / delta_t)  # 5760 if delta_t = 15s
    step_size = round(total_seconds_1_day / num_steps)
    complete_seconds = pd.Series(np.arange(0, total_seconds_1_day + 1, step_size))

    # Create a DataFrame with the complete range of seconds
    return pd.DataFrame({"seconds": complete_seconds[:-1]})  # drop the last second


def load_tec_data(
    year: int, datafolder: str | Path, load_with_polars=True
) -> pd.DataFrame:
    """
    Parameters
    ----------
    year : int
        Only the last two numbers are important.

    datafolder : str | Path
        The path of the folder where the year files are located.

    load_with_polars : bool
        if True then it uses polars library for reading files (~40% less time).

    Returns
    -------
    DataFrame : Pandas Dataframe with 'year', 'DOY', 'seconds' & 'TEC' columns.
    """

    year_files = get_tec_year_files(year, datafolder)

    # read year files
    DOY_index_start = -7
    DOY_index_end = -4
    year_df = pd.DataFrame()  # Initialize

    for year_file in year_files:
        day_df = read_and_process_tec_file(year_file, load_with_polars)
        DOY = int(year_file.name[DOY_index_start:DOY_index_end])
        day_df["DOY"] = DOY
        day_df["year"] = year

        year_df = pd.concat([year_df, day_df])

    # fill missing days (only if there are missing days)
    year_df = fill_missing_days(year_df)

    # sort
    year_df.sort_values(["DOY", "seconds"], inplace=True)
    year_df["datetime"] = create_datetimes(year_df)
    year_df.set_index("datetime", inplace=True)
    return year_df.loc[:,["tec"]]

def load_symh_data(filepath : str | Path, skiprows = 24) -> pd.DataFrame:
    """
    DEPRECATED FUNCTION
    Parameters
    ----------
    filepath : str | Path
        file to read SYM-H data.
    skiprows : int 
        Default 24

    Returns 
    -------
    DataFrame 
    """
    ASY_df = pd.read_csv(filepath, skiprows=skiprows, delim_whitespace=True)
    dt = pd.to_datetime(ASY_df.DATE + "-" + ASY_df.TIME)

    symh = ASY_df[["SYM-H"]].copy()
    symh.index = dt

    del ASY_df 
    gc.collect()
    return symh


def fill_missing_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing days in df.
    Only fills days in beetween.
    """
    DOYs = df.DOY.unique()

    if DOYs.max() - DOYs.min() == DOYs.size - 1:
        return df

    temp_df = df.copy()
    delta_t = int(abs(df.seconds.diff(1)).min())
    year = int(df.iloc[0, df.columns.get_loc("year")])
    for day in range(1, DOYs.max()):
        if not day in DOYs:
            df_day = create_seconds_df(delta_t)
            logging.info(
                f"Day {day} filled with NaNs for year {year} because it was missing."
            )
            df_day["DOY"] = day
            temp_df = pd.merge(temp_df, df_day, how="outer", on=["DOY", "seconds"])

    return temp_df


def DOY_to_datetime(year: int, DOY: int, seconds: int = 0) -> datetime:
    """
    Parameters
    ----------
    year : int
        the year is important because of leap years.

    DOY : int
        Day of the Year.

    seconds : Optional[int]

    Returns
    -------
    datetime

    """
    return datetime.strptime(f"{year}-{DOY}", "%Y-%j") + timedelta(0, seconds)


def create_datetimes(df: pd.DataFrame):
    """
    Create a DatetimeIndex with datetimes ranging from initial day to last day in df.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    DatetimeIndex
    """

    delta_t = int(abs(df.seconds.diff(1)).min())
    year = int(df.iloc[0, df.columns.get_loc("year")])
    DOY_start = int(df["DOY"].min())
    DOY_end = int(df["DOY"].max())
    seconds_start = int(df.loc[df["DOY"] == DOY_start, "seconds"].min())
    seconds_end = int(df.loc[df["DOY"] == DOY_end, "seconds"].max())

    start_date = DOY_to_datetime(year, DOY_start, seconds_start)
    end_date = DOY_to_datetime(year, DOY_end, seconds_end)

    return pd.date_range(start_date, end_date, freq=timedelta(0, delta_t))

def basic_stats(series : pd.Series) -> plt.Axes:
    """
    Imprimir Tamaño de la serie, porcentaje y número de NaNs.
    Plotear gráfica que muestre los NaNs a lo largo del tiempo.
    Plotear histograma.
    """
    n_nan = series.isna().sum()
    nan_ratio = n_nan/series.size
    print(f"Tamaño de la serie: {series.size}")
    print(f"Porcentaje de NaNs: {nan_ratio:.3%}")
    print(f"{n_nan} NaNs y {(series.size - n_nan)} Valores no nulos.")
    

    _, axs = plt.subplots(2)
    axs[0].plot(series.isna().astype(int))
    axs[0].set_yticks([0, 1], ["Number", "NaN"])

    axs[1].hist(series, bins=256)

    return axs

def load_symh_wdc(filepath : str | Path) -> pd.DataFrame:
    """
    Load SYM and ASY from file with WDC format.

    Parameters
    ----------
    filepath : str | Path
        plain text file with WDC format
    
    Returns
    -------
    DataFrame

    """
    columns = ["year", "month", "day", "component", "hour UT", "index", *[str(i) for i in range(1, 61)], "hour mean"]
    data = {c:[] for c in columns}

    filepath = Path(filepath)

    with filepath.open("r") as f:
        for line in f.readlines():
            data["year"].append(line[12:14])
            data["month"].append(line[14:16])
            data["day"].append(line[16:18])
            data["component"].append(line[18:19])
            data["hour UT"].append(line[19:21])
            data["index"].append(line[21:24])

            last_values = line[34:].split()
            for i, val in enumerate(last_values[:-1]):
                data[str(i+1)].append(val)
            data["hour mean"].append(last_values[-1])
    df = pd.DataFrame(data)

    symh_df = df.loc[(df["index"] == "SYM") & (df["component"] == "H")]
    
    #
    minutes = [i+1 for i in range(60)]
    all_symh = []
    all_minutes = []
    all_hours = []
    all_months = []
    all_years = []
    all_days = []

    # expand minutess

    for _, row in symh_df.iterrows():
        for min in minutes:
            all_minutes.append(min-1)
            all_symh.append(int(row[str(min)]))
            all_hours.append(int(row["hour UT"]))
            all_months.append(int(row["month"]))
            all_years.append(2000 + int(row["year"]))
            all_days.append(int(row["day"]))

    final_df = pd.DataFrame({"year" : all_years,
                             "month" : all_months,
                             "day" : all_days,
                             "hour" : all_hours,
                             "minute" : all_minutes,
                             "symh" : all_symh})
    
    # add datetimes
    delta_t = timedelta(0, 60)
    start_date = datetime(year=final_df.iloc[0]["year"], 
                        month=final_df.iloc[0]["month"],
                        day=final_df.iloc[0]["day"], 
                        hour=final_df.iloc[0]["hour"],
                        minute=final_df.iloc[0]["minute"])
    end_date = datetime(year=final_df.iloc[-1]["year"], 
                        month=final_df.iloc[-1]["month"],
                        day=final_df.iloc[-1]["day"], 
                        hour=final_df.iloc[-1]["hour"],
                        minute=final_df.iloc[-1]["minute"])
    dts = pd.date_range(start_date, end_date, freq=delta_t)
    final_df["datetime"] = dts
    final_df.set_index("datetime", inplace=True)
    return final_df.loc[:,["symh"]]


if __name__ == "__main__":
    year = 2000
    datafolder = Path("tucu/")
    df = load_tec_data(year, datafolder)

    DOYs = df.DOY.unique()
    print("DOYs size :", DOYs.size)
    print(DOYs.min(), DOYs.max())
    for day in range(1, DOYs.max()):
        if not day in DOYs:
            print("FALTA", day)

    print("len df", len(df))
    # print(df[df["DOY"] == 1])

    # make dts

    print(df[df["DOY"] == 125])
