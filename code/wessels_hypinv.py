import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
import os
from tqdm.auto import tqdm
from shapely import wkt
import xvec

def get_bounds_as_gdf(ds):
    """
    Given a NetCDF file with georeferenced raster data, returns a GeoDataFrame
    with a single bounding box geometry representing the dataset extent.
    
    Parameters:
        nc_path (str): Path to the NetCDF file.
    
    Returns:
        gpd.GeoDataFrame: A GeoDataFrame with one row containing the bounding box.
    """
    # Extract coordinate variables (commonly named 'x' and 'y')
    try:
        x = ds['x'].values
        y = ds['y'].values
    except KeyError:
        raise ValueError("Could not find coordinate variables 'x' and 'y' in the dataset.")

    # Compute bounds
    bounds = (x.min(), y.min(), x.max(), y.max())
    geom = box(*bounds)

    # Get CRS from dataset
    try:
        crs = ds.rio.crs
    except AttributeError:
        crs = None  # Will be set to None if CRS is missing

    return gpd.GeoDataFrame(index=[0], geometry=[geom], crs=crs).to_crs(epsg=4326)

def extract_points(ds, points, crs):
    """
    Extracts data values at specified points from a locally stored zarr dataset.

    Parameters:
    - ds: str, path to the zarr dataset or dataset.
    - points: GeoDataFrame, point locations to extract data.

    Returns:
    - DataFrame containing extracted data values and point indices in the same crs as ds.
    """

    if isinstance(ds, str):
        ds = xr.open_zarr(ds)
        ds.rio.set_crs(crs, inplace=True)
    
    # get the bounding box of the dataset
    geo = get_bounds_as_gdf(ds).to_crs(crs)

    # Reproject points to match the CRS of the dataset
    if points.crs != crs:
        points = points.to_crs(crs)

    # Clip the raw data to the bounding box
    points = points.clip(geo)
    print(f'got {points.shape[0]} point from {ds.title}')

    # Extract data at points
    extracted = ds.xvec.extract_points(
        points['geometry'], 
        x_coords="x", 
        y_coords="y", 
        index=True
    )
    
    return extracted

def xr_to_gdf(ds, crs=None):
    """
    Convert an xarray DataArray to a GeoDataFrame.
    
    Parameters:
        xr_data (xarray.DataArray): The xarray data to convert.
        crs (str or dict, optional): Coordinate reference system for the GeoDataFrame.
        
    Returns:
        gpd.GeoDataFrame: The converted GeoDataFrame.
    """
    df = ds.xvec.to_geodataframe(long=True).pivot_table(
    index=['geometry'],  # Replace with your actual spatial and other relevant dimensions
    columns='wavelength',
    values='reflectance'
    ).reset_index()
    df.columns = [str(col) if col != 'geometry' else col for col in df.columns]
    gdf = gpd.GeoDataFrame(df, geometry=df['geometry'], crs=crs).to_crs(epsg=4326)
    return gdf

if __name__ == "__main__":
    # Extract data for all points

    save_dir = "/mnt/hdd1/invasives/hypinvalimap/data"
    zarr_root = f"{save_dir}/2023_bioscape_invasives_tiles"

    files = [
        f"{zarr_root}/tile_{x}_{y}.zarr"
        for x in range(32, 36)
        for y in range(16, 22)
    ]

    ds = xr.open_dataset(r"/mnt/hdd1/invasives/hypinvalimap/data/AVIRIS-NG_BIOSCAPE_V02_L3_32_16_RFL.nc", engine="rasterio", chunks="auto")
    dscrs = ds.rio.crs

    joined = pd.read_csv(r'/home/geethen/invasives/hypinvalimap/data/Wessels_CS2_exp3_gdf23.csv')

    joined['geometry'] = joined['geometry'].apply(wkt.loads)
    joined = gpd.GeoDataFrame(joined, geometry='geometry', crs='EPSG:4326')


    for file in tqdm(files):
        # Extract tile name and construct save path in save_dir
        tile_name = os.path.basename(file).replace('.zarr', '.geojson')
        save_path = os.path.join(save_dir, "wessels_"+tile_name)
        
        if os.path.exists(save_path):
            print(f"Skipping {tile_name}, already exists.")
        else:
            print(f"Processing {tile_name}...")
            gdf = xr_to_gdf(extract_points(file, joined, crs= dscrs), dscrs)
            print(gdf.shape)
            
            gdf.to_file(save_path)
    
    save_dir = "/mnt/hdd1/invasives/hypinvalimap/data"

    # List of GeoJSON file paths
    files = [
        os.path.join(save_dir, f"wessels_tile_{x}_{y}.geojson")
        for x in range(32, 36)
        for y in range(16, 22)
    ]

    # Combine all GeoJSONs into a single GeoDataFrame
    gdfs = []
    for file in tqdm(files):
        gdf = gpd.read_file(file).set_crs(epsg=4326, allow_override=True)
        gdfs.append(gdf)

    xdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs='EPSG:4326')
    
    sjoined = gpd.sjoin_nearest(joined[['fid', 'class', 'group','change', 'notes', 'geometry']], xdf, how='inner', distance_col='dist')
    sjoined.to_file(r"/home/geethen/invasives/hypinvalimap/data/2023_wessels_extracted.geojson")