from pydantic_ai import Agent
import os
import requests
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from requests.models import PreparedRequest
from urllib.request import Request, urlopen
import numpy as np
from dateutil.relativedelta import relativedelta
from typing import Optional
import mpld3

import json
import plotly.express as px
import pandas as pd
from pydantic import Field
import pytz
from enum import Enum

from typing import Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pytz
from pydantic import BaseModel

from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pytz
import requests
import pandas as pd
import plotly.express as px
from pydantic import BaseModel
import json
from rasterio.io import MemoryFile
import re


hcdp_api_token = json.load(open("tokens.json"))["hcdp_api_token"]
gmaps_api_key = json.load(open("tokens.json"))["gmaps_api_key"]

# Please input your email address. This will be used for user logging or distributing data packages
# email = "INSERT_EMAIL_ADDRESS_HERE"

api_base_url = json.load(open("tokens.json"))["api_base_url"]
# Setup header for API requests
header = {
"Authorization": f"Bearer {hcdp_api_token}"
}
output_dir = "output_files"


class DataType(str, Enum):
    TEMPERATURE = "temperature"
    RAINFALL = "rainfall"
    RELATIVE_HUMIDITY = "relative_humidity"
    NDVI_MODIS = "ndvi_modis"
    IGNITION_PROBABILITY = "ignition_probability"

class Aggregation(str, Enum):
    MIN = "min"
    MAX = "max"
    MEAN = "mean"

class Production(str, Enum):
    '''
    Production can be "new" or "legacy". Legacy rainfall maps are available from 1920-2012, whereas new rainfall maps are available from 1990-present
    '''
    NEW = "new"
    LEGACY = "legacy"

class Period(str, Enum):
    '''
    This is the resolution of the datapoints. NDVI_MODIS only has a DAY resolution. If user asks for year or month for NDVI, return DAY
    '''
    
    DAY = "day"
    MONTH = "month"
    YEAR = "year"

class Extent(str, Enum):
    '''
    HAWAII = Big Island, Hawai'i
    KAUAI = Kauai island, Hawai'i
    HONOLULU = O'ahu island, Hawai'i
    MLM = Moloka'i and Lanai islands, Maui island, 
    STATEWIDE = any time latitude and longitude coordinates are input, extent = statewide    
    '''
    STATEWIDE = "statewide"  # Data for the whole state
    HAWAII = "bi"            # Hawaii county
    KAUAI = "ka"             # Kauai county
    MLM = "mn"              # Maui, Molokai, Lanai county
    HONOLULU = "oa"          # Honolulu county

class ClimateDataParams(BaseModel):
    datatype: DataType
    period: Period
    start: str
    end: str
    extent: Extent
    lat: Optional[float] = None
    lng: Optional[float] = None
    # Optional fields that differ between temperature and rainfall
    aggregation: Optional[Aggregation] = None
    production: Optional[Production] = None


class ClimateAPI:     
    def __init__(self, api_base_url: str, header: Dict[str, str]):         
        self.api_base_url = api_base_url         
        self.header = header         
        self.raster_timeseries_ep = "/raster/timeseries"              

    def get_timeseries_data(self, params: ClimateDataParams) -> pd.DataFrame:         
        """Get timeseries data from the API based on provided parameters"""         
        url = f"{self.api_base_url}{self.raster_timeseries_ep}"                  
        params_dict = params.model_dump()
        
        # Remove None values from params
        params_dict = {k: v for k, v in params_dict.items() if v is not None}
                  
        res = requests.get(url, params_dict, headers=self.header)         
        res.raise_for_status()         
        print(f"Constructed API request URL: {res.url}")                  
        data = res.json()         
        df_data = list(data.items())                  
        
        # Determine value column name based on datatype
        if params.datatype == DataType.TEMPERATURE:
            value_col = f"{params.aggregation.value.capitalize()} {params.datatype.value.capitalize()} (°C)"
        else:  # For rainfall
            value_col = f"{params.datatype.value.capitalize()} (mm)"
                  
        df = pd.DataFrame(df_data, columns=["Date", value_col])         
        df = df.sort_values(by="Date")                  
        return df   

    def plot_timeseries(self, df: pd.DataFrame, params: ClimateDataParams) -> None:
        """Line plot of timeseries data using Plotly. This is specifically for plotting a variable over time. Always output HTML unless specifically told not to."""
        value_col = df.columns[1]  # Second column contains the values
        
        # Create appropriate title based on params
        if params.datatype == DataType.TEMPERATURE:
            title = f"Summary of {params.aggregation.value} {params.datatype.value} from {params.start} to {params.end}"
        else:
            title = f"Summary of {params.datatype.value} from {params.start} to {params.end}"
            
        if params.lat is not None and params.lng is not None:
            title += f" for location Latitude: {params.lat}, Longitude: {params.lng}"
            
        fig = px.line(df, title=title, x="Date", y=value_col)
        fig.write_html(f"{output_dir}/ndvis.html")
        return fig
    

hcdp_info = {
    'visualize data tutorial': 'https://www.hawaii.edu/climate-data-portal/visualize-data-tutorial/',
    'explore station data tutorial':'https://www.hawaii.edu/climate-data-portal/visualize-data-tutorial-explore-station-data/', 
    'export data tutorial':'https://www.hawaii.edu/climate-data-portal/export-data-tutorial/',
    'rainfall mapping':'https://www.hawaii.edu/climate-data-portal/rainfall-mapping-history/',
    'climate monitioring history':'https://www.hawaii.edu/climate-data-portal/climate-monitoring-history/',
    'publications': ['https://www.hawaii.edu/climate-data-portal/publications-list/', 'https://www.hawaii.edu/climate-data-portal/research-highlights/'],
    'how to cite':'https://www.hawaii.edu/climate-data-portal/how-to-cite-3/',
    'acknowledgements':'https://www.hawaii.edu/climate-data-portal/acknowledgements/',
    'HCDP history':'https://www.hawaii.edu/climate-data-portal/2339-2/',
    'team':'https://www.hawaii.edu/climate-data-portal/team/',
    'twitter':'https://x.com/hiclimateportal',
    'email':'hcdp@hawaii.edu',
    'rainfall atlas GIS raster':'https://www.hawaii.edu/climate-data-portal/rainfall-atlas/',
    'evapotranspiration GIS raster':'https://www.hawaii.edu/climate-data-portal/evapotranspiration-atlas/',
    'solar radiation GIS raster':'https://www.hawaii.edu/climate-data-portal/solar-radiation-atlas/',
    'climate of hawaii GIS raster':'https://www.hawaii.edu/climate-data-portal/climate-atlas/',
    'grid documentation GIS raster':'https://atlas.uhtapis.org/evapo/assets/files/PDF/Metadata_Grids_Climate.pdf',
    'climate tools':'https://www.hawaii.edu/climate-data-portal/climate-tools-2/',
    'cultural resources':'https://www.hawaii.edu/climate-data-portal/news/'
}


today = datetime.now(pytz.timezone("US/Hawaii"))

prompt_process_query = f"""You are a Concierge AI assistant for the Hawai'i Climate Data Portal (HCDP). Your primary role is to provide accurate, data-driven answers using the available tools.

Users typically ask two types of queries - Specific Climate Data Queries and General Information Queries.
Your task is to carefully classify the query into one of these two categories and respond accordingly.
During the classification process, you should answer the following questions:
1. What is the user exactly asking for?
2. Does the query want a plot to be shown?
3. Which tool should be used to answer the requirements in the query?

The 2 types of queries are:

1. Specific Climate Data Queries
Queries seeking climate data for a specified location and time period. Use the HCDP API to fetch accurate data.
These types of queries may also ask for a plot, a summary of the data, of to show or display change in raster or a variable over time. 


Examples:
"What's the average temperature in Honolulu for last month?"
"What was the maximum rainfall in Kauai last year?"
"Plot the temparature change over the last 30 days in Hilo."
"Show the rainfall change over the last year in Honolulu."

When receiving this type of query, classify it clearly and fill the following template to fetch data:
datatype: "temperature" | "rainfall | relative_humidity | ndvi_modis | ignition_probability",
aggregation: "min" | "max" | "mean",
period: "day" | "month" | "year",
location name: str,
start_date: Optional[str],
end_date: Optional[str]

Note: today's date is ({today}). Use it to correctly identify start_date and end_date.


2. General Information Queries
Queries about HCDP's purpose, functionalities, or how to access and use its data.

Examples:
"Where can I find rainfall data?"
"How do I visualize temperature data?"
"What research has been conducted using HCDP data?"
"How do I export or download data from HCDP?"

For these queries, rely strictly on the provided HCDP information below-{hcdp_info} 
If the information is unavailable, inform the user accordingly.


Response Format:
Always respond clearly using concise HTML code. Only if relevant, provide a plot link using anchor tags with target="_blank".:
<p>Your concise answer here.</p>
<a href="/plot" target="_blank">Open Plot</a> <!-- include only if user asks for a display/show/plot -->


If a user's query isn't clear or doesn't fit these categories, politely ask to rewrite the query more clearly.

"""


prompt_process_agent = Agent(  
        "gemini-2.5-pro-exp-03-25",
        result_type=str,    
        system_prompt=prompt_process_query,
        model_settings={
            'temperature': 0.0,
            'api_key':json.load(open("tokens.json"))["gemini_api_key"]}    
    )


@prompt_process_agent.tool_plain  
def get_temperature_timeseries(     
    aggregation: Aggregation,     
    period: Period,     
    lat: float,     
    lng: float,     
    start_date: Optional[str] = None,      
    end_date: Optional[str] = None,
    extent: Optional[Extent] = Extent.STATEWIDE
) -> Dict[str, Any]:     
    """Return temperature timeseries data for the specified location, period and aggregation"""     
    api = ClimateAPI(api_base_url=api_base_url, header=header)     
    print("API initialized:", api)          

    today = datetime.now(pytz.timezone("US/Hawaii"))     
    yesterday = today - timedelta(days=1)     
    previous_year = today - relativedelta(years=1)          

    start_str = start_date if start_date else previous_year.strftime("%Y-%m-%d")     
    end_str = end_date if end_date else yesterday.strftime("%Y-%m-%d")          

    # Create params for temperature (requires aggregation)
    params = ClimateDataParams(         
        datatype=DataType.TEMPERATURE,         
        aggregation=aggregation,         
        period=period,         
        start=start_str,         
        end=end_str,         
        # extent=extent,     
        extent="statewide",   
        lat=lat,         
        lng=lng     
    )          

    print("Query parameters:", params)          

    df = api.get_timeseries_data(params)          

    # Return structured result with data preview + summary     
    result = {         
        "data_preview": df.head(5).to_dict(orient="records") + df.tail(5).to_dict(orient="records"),
        "summary": {             
            "mean": df.iloc[:, 1].mean(),             
            "min": df.iloc[:, 1].min(),             
            "max": df.iloc[:, 1].max(),             
            "location": {"lat": lat, "lng": lng},             
            "period": f"{start_str} to {end_str}"         
        }     
    }

    # Plot the time series if data exists and has a "Date" column
    if not df.empty and "Date" in df.columns:
        api.plot_timeseries(df, params)
        print("Time series plot has been generated.")
    else:
        print("No time series data available for plotting.")
     
    return result  

@prompt_process_agent.tool_plain  
def get_rainfall_timeseries(     
    period: Period,     
    lat: float,     
    lng: float,     
    start_date: Optional[str] = None,      
    end_date: Optional[str] = None,
    production: Production = Production.NEW,
    extent: Optional[Extent] = Extent.STATEWIDE
) -> Dict[str, Any]:     
    """Return the max, mean and min of the rainfall data. Keep it concise."""     
    api = ClimateAPI(api_base_url=api_base_url, header=header)     
    print("API initialized:", api)          

    today = datetime.now(pytz.timezone("US/Hawaii"))     
    yesterday = today - timedelta(days=1)     
    previous_year = today - relativedelta(years=1)          

    start_str = start_date if start_date else previous_year.strftime("%Y-%m-%d")     
    end_str = end_date if end_date else yesterday.strftime("%Y-%m-%d")          

    # Create params for rainfall (requires production instead of aggregation)
    params = ClimateDataParams(         
        datatype=DataType.RAINFALL,         
        production="new",         
        period=period, 
        start = start_str,
        end = end_str,
        extent="statewide",    
        lat=lat,         
        lng=lng     
    )      
    
    print("Query parameters:", params)          

    df = api.get_timeseries_data(params)          

    # Return structured result with data preview + summary     
    result = {         
        "data_preview": df.head(5).to_dict(orient="records") + df.tail(5).to_dict(orient="records"),
        "summary": {             
            "mean": df.iloc[:, 1].mean(),             
            "min": df.iloc[:, 1].min(),             
            "max": df.iloc[:, 1].max(),             
            "location": {"lat": lat, "lng": lng},             
            "period": f"{start_str} to {end_str}"         
        }     
    }

    # Plot the time series if data exists and has a "Date" column
    if not df.empty and "Date" in df.columns:
        api.plot_timeseries(df, params)
        print("Time series plot has been generated.")
    else:
        print("No time series data available for plotting.")
     
    return result


@prompt_process_agent.tool_plain  
def get_relative_humidity_timeseries(     
    period: Period,     
    lat: float,     
    lng: float,     
    start_date: Optional[str] = None,      
    end_date: Optional[str] = None,
    # production: Production = Production.NEW,
    extent: Optional[Extent] = Extent.STATEWIDE
) -> Dict[str, Any]:     
    """Return the max, mean and min relative humidity for the queried location. Keep it concise."""     
    api = ClimateAPI(api_base_url=api_base_url, header=header)     
    print("API initialized:", api)          

    today = datetime.now(pytz.timezone("US/Hawaii"))     
    yesterday = today - timedelta(days=1)     
    previous_year = today - relativedelta(years=1)          

    start_str = start_date if start_date else previous_year.strftime("%Y-%m-%d")     
    end_str = end_date if end_date else yesterday.strftime("%Y-%m-%d")          

    # Create params for rainfall (requires production instead of aggregation)
    params = ClimateDataParams(         
        datatype=DataType.RELATIVE_HUMIDITY,         
        # production="new",         
        period="day", 
        start = start_str,
        end = end_str,
        extent="statewide",    
        lat=lat,         
        lng=lng     
    )      
    
    print("Query parameters:", params)          

    df = api.get_timeseries_data(params)          

    # Return structured result with data preview + summary     
    result = {         
        "data_preview": df.head(5).to_dict(orient="records") + df.tail(5).to_dict(orient="records"),
        "summary": {             
            "mean": df.iloc[:, 1].mean(),             
            "min": df.iloc[:, 1].min(),             
            "max": df.iloc[:, 1].max(),             
            "location": {"lat": lat, "lng": lng},             
            "period": f"{start_str} to {end_str}"         
        }     
    }

    # Plot the time series if data exists and has a "Date" column
    if not df.empty and "Date" in df.columns:
        api.plot_timeseries(df, params)
        print("Time series plot has been generated.")
    else:
        print("No time series data available for plotting.")
     
    return result


@prompt_process_agent.tool_plain
def get_lat_long_from_location(location_name:str):
    """
    Retrieves the latitude and longitude for a given location name using the Google Geocoding API.
    
    Parameters:
        api_key (str): Your Google Maps API key.
        location_name (str): The name or address of the location.
        
    Returns:
        dict: A dictionary with 'lat' and 'lng' keys if successful, or None otherwise.
    """
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": location_name,
        "key": json.load(open("tokens.json"))["gmaps_api_key"]
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes.
        data = response.json()
        if data.get("status") == "OK" and data.get("results"):
            location = data["results"][0]["geometry"]["location"]
            return location
        else:
            print("Error: Could not retrieve geocode data for location:", location_name)
            return None
    except Exception as e:
        print("Error retrieving geocode data:", e)
        return None


@prompt_process_agent.tool_plain  
def get_vegetation_data_timeseries(     
    period: Period,     
    lat: float,     
    lng: float,     
    start_date: Optional[str] = None,      
    end_date: Optional[str] = None,
    # production: Production = Production.NEW,
    extent: Optional[Extent] = Extent.STATEWIDE
) -> Dict[str, Any]:     
    """NDVI is an index quanitying vegetation health and density. Return the max, mean and min NDVI number for the query. Keep it concise."""     
    api = ClimateAPI(api_base_url=api_base_url, header=header)     
    print("API initialized:", api)          

    today = datetime.now(pytz.timezone("US/Hawaii"))     
    yesterday = today - timedelta(days=1)     
    previous_year = today - relativedelta(years=1)          

    start_str = start_date if start_date else previous_year.strftime("%Y-%m-%d")     
    end_str = end_date if end_date else yesterday.strftime("%Y-%m-%d")          

    # Create params for rainfall (requires production instead of aggregation)
    params = ClimateDataParams(         
        datatype="ndvi_modis",         
        # production="new",         
        period=period, 
        start = start_str,
        end = end_str,
        extent="statewide",    
        lat=lat,         
        lng=lng     
    )      
    
    print("Query parameters:", params)          

    df = api.get_timeseries_data(params)          

    # Return structured result with data preview + summary     
    result = {         
        "data_preview": df.head(5).to_dict(orient="records") + df.tail(5).to_dict(orient="records"),
        "summary": {             
            "mean": df.iloc[:, 1].mean(),             
            "min": df.iloc[:, 1].min(),             
            "max": df.iloc[:, 1].max(),             
            "location": {"lat": lat, "lng": lng},             
            "period": f"{start_str} to {end_str}"         
        }     
    }

    # Plot the time series if data exists and has a "Date" column
    if not df.empty and "Date" in df.columns:
        api.plot_timeseries(df, params)
        print("Time series plot has been generated.")
    else:
        print("No time series data available for plotting.")
     
    return result


@prompt_process_agent.tool_plain  
def get_ignition_probability_timeseries(     
    period: str,     
    lat: float,     
    lng: float,     
    start_date: Optional[str] = None,      
    end_date: Optional[str] = None,
    # production: Production = Production.NEW,
    extent: Optional[Extent] = Extent.STATEWIDE
) -> Dict[str, Any]:     
    """The ignition probability product shows the daily probability of large (8+ acre) fire ignitions based on current and past climate. Keep it concise."""     
    api = ClimateAPI(api_base_url=api_base_url, header=header)     
    print("API initialized:", api)          

    today = datetime.now(pytz.timezone("US/Hawaii"))     
    yesterday = today - timedelta(days=1)     
    previous_year = today - relativedelta(years=1)          

    start_str = start_date if start_date else previous_year.strftime("%Y-%m-%d")     
    end_str = end_date if end_date else yesterday.strftime("%Y-%m-%d")          

    # Create params for rainfall (requires production instead of aggregation)
    params = ClimateDataParams(         
        datatype="ignition_probability",         
        # production="new",         
        period="day", 
        start = start_str,
        end = end_str,
        extent="statewide",    
        lat=lat,         
        lng=lng     
    )      
    
    print("Query parameters:", params)          

    df = api.get_timeseries_data(params)          

    # Return structured result with data preview + summary     
    result = {         
        "data_preview": df.head(5).to_dict(orient="records") + df.tail(5).to_dict(orient="records"),
        "summary": {             
            "mean": df.iloc[:, 1].mean(),             
            "min": df.iloc[:, 1].min(),             
            "max": df.iloc[:, 1].max(),             
            "location": {"lat": lat, "lng": lng},             
            "period": f"{start_str} to {end_str}"         
        }     
    }

    # Plot the time series if data exists and has a "Date" column
    if not df.empty and "Date" in df.columns:
        api.plot_timeseries(df, params)
        print("Time series plot has been generated.")
    else:
        print("No time series data available for plotting.")
     
    return result


## Raster Rendering Code
def display_raster(params, title, cmap=plt.cm.viridis.reversed(), nodata_color="#f0f0f0"):
    raster_ep = "/raster"
    url = f"{api_base_url}{raster_ep}"
    url_constructor = PreparedRequest()
    url_constructor.prepare_url(url, params)
    full_url = url_constructor.url
    print(f"Constructed API request URL: {full_url}")

    req = Request(full_url, headers=header)

    # Read raster data from API
    try:
        with urlopen(req) as raster:
            with MemoryFile(raster.read()) as memfile:
                with memfile.open() as dataset:
                    data = dataset.read(1)  # Read first band
                    nodata = dataset.nodata or data[0, 0]
                    masked = np.ma.masked_equal(data, nodata)

        # Set colormap nodata color
        cmap.set_bad(nodata_color)

        # Plot
        fig, ax = plt.subplots(figsize=(20, 10), facecolor="#e0e0e0")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=20)
        im = ax.imshow(masked, cmap=cmap)
        fig.colorbar(im, ax=ax)

        # Save interactive HTML with mpld3
        html_str = mpld3.fig_to_html(fig)
        with open(f"{output_dir}/ndvis.html", "w") as f:
            f.write(html_str)

    except Exception as e:
        print(f"Error displaying raster: {e}")

@prompt_process_agent.tool_plain
def display_rainfall_raster(
    date: str,
    production: Production = Production.NEW,
    extent: Extent = Extent.STATEWIDE,
    period: Period = Period.MONTH
) -> None:
    """Display the rainfall raster for the given date and extent."""
    params = {
        "datatype": "rainfall",
        "production": production.value,  # Use `.value` if Production is an Enum
        "period": "month",
        "date": date,
        "extent": extent.value  # Use `.value` if Extent is an Enum
    }

    title = f"Rainfall Raster for {extent.value.title()} on {date}"
    display_raster(params, title)
    
@prompt_process_agent.tool_plain
def display_temperature_raster(
    date: str,
    aggregation: Aggregation = Aggregation.MEAN,
    extent: Extent = Extent.STATEWIDE,
    period: Period = Period.MONTH
) -> None:
    """Display the temperature raster for the given date and extent."""
    params = {
        "datatype": "temperature",
        "aggregation": aggregation.value,  # Use `.value` if Production is an Enum
        "period": "month",
        "date": date,
        "extent": extent.value  # Use `.value` if Extent is an Enum
    }

    title = f"Temperature ({aggregation.value.title()}) Raster for {extent.value.title()} on {date}"
    display_raster(params, title)


@prompt_process_agent.tool_plain
def display_relative_humidity_raster(
    date: str,
    # aggregation: Aggregation = Aggregation.MEAN,
    extent: Extent = Extent.STATEWIDE
) -> None:
    """Display the relative humidity raster for the given date and extent."""
    params = {
        "datatype": "relative_humidity",
        # "aggregation": aggregation.value,  # Use `.value` if Production is an Enum
        "period": "day",
        "date": date,
        "extent": extent.value  # Use `.value` if Extent is an Enum
    }

    title = f"Relative Humidity Raster for {extent.value.title()} on {date}"
    display_raster(params, title)


@prompt_process_agent.tool_plain
def display_vegetation_cover_raster(
    date: str,
    # aggregation: Aggregation = Aggregation.MEAN,
    extent: Extent = Extent.STATEWIDE
) -> None:
    """Display the vegetation cover NDVI raster for the given date and extent."""
    params = {
        "datatype": "ndvi_modis",
        # "aggregation": aggregation.value,  # Use `.value` if Production is an Enum
        "period": "day",
        "date": date,
        "extent": extent  # Use `.value` if Extent is an Enum
    }

    title = f"Normalized Difference Vegetation Index (NDVI) Raster for {extent.value.title()} on {date}"
    display_raster(params, title)


@prompt_process_agent.tool_plain
def display_ignition_probability_raster(
    date: str,
    # aggregation: Aggregation = Aggregation.MEAN,
    extent: Extent = Extent.STATEWIDE
) -> None:
    """Display the ignition probability raster for the given date and extent."""
    params = {
        "datatype": "ignition_probability",
        # "aggregation": aggregation.value,  # Use `.value` if Production is an Enum
        "period": "day",
        "date": date,
        "extent": extent  # Use `.value` if Extent is an Enum
    }

    title = f"Ignition Probability Raster for {extent.value.title()} on {date}"
    display_raster(params, title)



def convert_temp(sentence):
    # Define a function to handle the replacement
    def repl(match):
        celsius_str = match.group(1)
        celsius_val = float(celsius_str)
        fahrenheit_val = celsius_val * 9/5 + 32
        # Format the Fahrenheit conversion to 2 decimal places and include both units.
        return f"{fahrenheit_val:.2f}°F ({celsius_str}°C)"
    
    # Use regex to search for a pattern matching a number (with optional decimal) followed by °C.
    new_sentence = re.sub(r"(-?\d+(?:\.\d+)?)°C", repl, sentence)
    return new_sentence


async def process_query(query: str) -> str:
    r = await prompt_process_agent.run(query)
    if r.data.startswith("```html"):
        r.data = r.data.replace("```html", "").replace("```", "")
    if r.data.endswith("```"):
        r.data = r.data.replace("```", "")
    
    return convert_temp(r.data.strip())