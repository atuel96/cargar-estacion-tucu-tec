import numpy as np
import pandas as pd
from pathlib import Path
import polars as pl

import logging

print()


def get_year_files(year: int, datafolder: str | Path):
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


def read_and_process_file(
    filepath: str | Path,
    load_with_polars=True,
):
    """
    Parameters
    ----------
    filepath : str | Path
        Path of the file.

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
        df.columns = ["seconds", "alpha", "TEC"]
    else:
        df = pd.read_csv(
            filepath,
            sep=" ",
            header=None,
            usecols=[0, 1, 7],
            names=["seconds", "alpha", "TEC"],
        )
    df_vTEC = df.query("alpha == 'Z00'")  # take just vTEC

    # Delete duplicates
    n_duplicated_values = df_vTEC.duplicated().sum()
    if n_duplicated_values:
        logging.info(f"{n_duplicated_values} duplicated values inf file {filepath}.")

    df_vTEC = df_vTEC.loc[:, ["seconds", "TEC"]].drop_duplicates()

    # Look for negative TEC values
    # negative_TEC = df_vTEC["TEC"][df_vTEC["TEC"] < 0]
    negative_TEC = df_vTEC.loc[df_vTEC["TEC"] < 0, "TEC"]
    negative_values = negative_TEC.count()
    if negative_values:
        df_vTEC.loc[df_vTEC["TEC"] < 0, "TEC"] = None
        logging.info(
            f"{negative_values} TEC negative values were replaced by NaNs for file {filepath}."
        )  # Info

    # frequency
    delta_t = int(df_vTEC.diff(1)["seconds"].mode()[0])

    # Fill non-existing values with Nan
    total_seconds_1_day = 86400  # 1 day in seconds
    num_steps = round(total_seconds_1_day / delta_t)  # 5760 if delta_t = 15s
    step_size = round(total_seconds_1_day / num_steps)
    complete_seconds = pd.Series(np.arange(0, total_seconds_1_day + 1, step_size))

    # Create a DataFrame with the complete range of seconds
    complete_df = pd.DataFrame(
        {"seconds": complete_seconds[:-1]}  # drop the last second
    )
    merged_df = pd.merge(complete_df, df_vTEC, on="seconds", how="left")
    merged_df.sort_values("seconds", inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    return merged_df[["seconds", "TEC"]]


def load_year_data(
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

    year_files = get_year_files(year, datafolder)

    # read year files
    DOY_index_start = -7
    DOY_index_end = -4
    year_df = pd.DataFrame()  # Initialize

    for year_file in year_files:
        day_df = read_and_process_file(year_file, load_with_polars)
        DOY = int(year_file.name[DOY_index_start:DOY_index_end])
        day_df["DOY"] = DOY
        day_df["Year"] = year

        year_df = pd.concat([year_df, day_df])
    return year_df


if __name__ == "__main__":
    year = 2000
    datafolder = Path("tucu/")
    df = load_year_data(year, datafolder)

    DOYs = df.DOY.unique()
    print("DOYs size :", DOYs.size)
    print(DOYs.min(), DOYs.max())
    for day in range(1, DOYs.max()):
        if not day in DOYs:
            print(day)

    # print(df[df["DOY"] == 125])
