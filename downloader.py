#!/bin/env python
from pathlib import Path
from datetime import datetime

import requests


class DownloadFailed(Exception):
    def __init__(self, *args):
        super().__init__(*args)


def download_asym_year(year):
    return download_asym(
        datetime(year, 1, 1),
        datetime(year, 12, 31, 23, 59),
        filename=f"ASY-SYM-WDCformat-{year}.dat",
    )


def download_asym(start_date, end_date, folderpath=Path(""), filename=None):
    year, month, day, hour, minute = start_date.strftime("%Y-%m-%d-%H-%M").split("-")
    duration = end_date - start_date
    dur_days = str(duration.days).zfill(3)
    dur_hours = str(duration.seconds // 3600).zfill(2)
    dur_minutes = str((duration.seconds % 3600) // 60).zfill(2)

    url = (
        f"https://wdc.kugi.kyoto-u.ac.jp/cgi-bin/aeasy-cgi?Tens={year[:3]}&Year={year[-1]}&"
        f"Month={month}&Day_Tens={day[0]}&Days={day[1]}&Hour={hour}&min={minute}&"
        f"Dur_Day_Tens={dur_days[:2]}0&Dur_Day={dur_days[-1]}&"
        f"Dur_Hour={dur_hours}&Dur_Min={dur_minutes}&Image+Type=GIF&COLOR=COLOR&AE+Sensitivity=0&"
        "ASY%2FSYM++Sensitivity=0&Output=ASY&Out+format=WDC"
    )

    print("Starting Download ...")
    response = requests.get(url)
    if response.status_code != 200:
        raise DownloadFailed("Download Failed, check url")

    filename = (
        filename
        or f"ASY-SYM-WDCformat-{start_date.strftime('%Y-%m-%d')}-{start_date.strftime('%Y-%m-%d')}.dat"
    )
    with open(
        Path(folderpath) / filename,
        "w",
    ) as f:
        f.write(response.text)

    print(
        "Data soruce: WDC for Geomagnetism, Kyoto. Visit: https://wdc.kugi.kyoto-u.ac.jp/\n\n"
    )
    print(
        "World Data Center for Geomagnetism, Kyoto, M. Nose, T. Iyemori, M. Sugiura, T. Kamei (2015), Geomagnetic AE index, doi:10.17593/15031-54800,\n"
        "World Data Center for Geomagnetism, Kyoto, M. Nose, T. Iyemori, M. Sugiura, T. Kamei (2015), Geomagnetic Dst index, doi:10.17593/14515-74000,\n"
        "and World Data Center for Geomagnetism, Kyoto, S. Imajo, A. Matsuoka, H. Toh, and T. Iyemori (2022), Mid-latitude Geomagnetic Indices ASY and"
        "SYM (ASY/SYM Indices), doi:10.14989/267216."
    )
    return response.text


if __name__ == "__main__":

    years = []
    for year in years:
        data = download_asym_year(year)