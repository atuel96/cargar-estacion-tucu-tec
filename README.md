# cargar-estacion-tucu-tec
Scripts y notebooks para cargar datos TEC de la estación TUCU


# API example

``` python
from downloader import download_asym_year
from data_utils import load_symh_wdc, load_tec_data

# Download 1 year of ASYM
year = 2000
download_asym_year(year) # file ASY-SYM-WDCformat-<year>.dat created

# Load only SYMH 
symh_year = load_symh_wdc(f"ASY-SYM-WDCformat-{year}.dat")

# TEC
tec_year = load_tec_data(year, "data")

# Merge both
df = pd.merge(left=tec_year, 
         right=symh_year, 
         right_on=symh_year.index, 
         how="left",left_index=True).loc[:,["tec", "symh"]]
```