import sys
sys.path.insert(1, '/bask/homes/f/fedu7800/vjgo8416-demoland/spatial_signatures/eo/ai_pipeline')

import pipeline
from pipeline import GeoTileDataset, read_data, plot_examples

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import geopandas as gpd

# Run the pipeline
pipeline.spatial_sig_prediction(
    geo_path= "/bask/homes/f/fedu7800/vjgo8416-demoland/spatial_signatures/grid_clean.geojson",
    vrt_file= "/bask/homes/f/fedu7800/vjgo8416-demoland/satellite_demoland/data/mosaic_cube/vrt_allbands/2017_combined.vrt",
    xgb_weights = "/bask/homes/f/fedu7800/vjgo8416-demoland/spatial_signatures/predictions/xgb_model_25_latlonh6_feb25_weighted.bin",
    model_weights = "/bask/homes/f/fedu7800/vjgo8416-demoland/satellite_demoland/models/satlas/weights/satlas-model-v1-lowres.pth",
    output_path= "/bask/homes/f/fedu7800/vjgo8416-demoland/satellite_demoland/data/2017_predictions.parquet",
    h3_resolution=6
)
