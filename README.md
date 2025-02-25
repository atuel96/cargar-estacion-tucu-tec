# cargar-estacion-tucu-tec
Scripts y notebooks para cargar datos TEC de la estación TUCU

# Create Environment

Use your prefered tool to create and set your environment (conda, venv, virtualenv, etc.)

Conda example:

``` bash
conda create --name tucudata python
conda activate tucudata
```

venv example:

``` bash
python -m venv .venv
source .venv/bin/activate
```

then, with the environment activated:

``` bash 
pip install -r requirements.txt
```
# Basic Usage Example

``` python
import pandas as pd

from downloader import download_asym_year
from data_utils import load_symh_wdc, load_tec_data

# Download 1 year of ASYM
year = 2000
download_asym_year(year) # file ASY-SYM-WDCformat-<year>.dat created

# Load only SYMH 
symh_year = load_symh_wdc(f"ASY-SYM-WDCformat-{year}.dat")

# TEC
tec_year = load_tec_data(year, "data")

# Resample TEC & interpolate
tec_year_resampled = (
    tec_year.resample(rule="1min").asfreq().interpolate(method="linear", limit=10)
)  # resample to 1 min & interpolate gaps shorter than 10 minutes


# Merge both
df = pd.merge(left=tec_year_resampled, 
         right=symh_year, 
         right_on=symh_year.index, 
         how="left",left_index=True).loc[:,["tec", "symh"]]
```