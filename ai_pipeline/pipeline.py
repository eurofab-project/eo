import torch
import rasterio
import torchvision
import geopandas as gpd
import numpy as np
import collections
import xgboost as xgb
import pandas as pd
import h3.api.numpy_int as h3
from rasterio.mask import mask

from torchvision.ops import FeaturePyramidNetwork
from torchvision.transforms.functional import resize

from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

from sklearn.preprocessing import LabelEncoder
from pyproj import Transformer
import matplotlib.pyplot as plt


class GeoTileDataset(Dataset):
    def __init__(self, geo_path, vrt_file, normalization_ranges=None, resize_shape=(3, 25, 25)):
        """
        Dataset for loading geospatial tiles from a GeoJSON or GeoParquet file.
        """
        self.geo_path = geo_path
        self.vrt_file = vrt_file
        self.normalization_ranges = normalization_ranges
        self.resize_shape = resize_shape  # Shape to resize to (channels, height, width)
        self.grid = self._load_geospatial_data()

    def _load_geospatial_data(self):
        if self.geo_path.endswith(".geojson"):
            return gpd.read_file(self.geo_path)
        elif self.geo_path.endswith(".parquet"):
            return gpd.read_parquet(self.geo_path)
        else:
            raise ValueError("Unsupported file format. Provide a GeoJSON or GeoParquet file.")

    def __len__(self):
        return len(self.grid)

    def __getitem__(self, idx):
        tile = self.grid.iloc[idx]
        
        # Extract tile geometry
        geom = [tile.geometry]
        
        # Read raster data
        with rasterio.open(self.vrt_file) as src:
            raster, _ = mask(src, geom, crop=True, all_touched=True)
        
        raster = raster[:3, :, :]  # Keep first 3 bands

        # Resize raster to the target shape
        raster = torch.tensor(raster, dtype=torch.float32)  # Convert to a tensor
        raster = resize(raster, size=self.resize_shape[1:])  # Resize height and width

        # Normalize raster if ranges are provided
        if self.normalization_ranges:
            for i in range(min(raster.shape[0], len(self.normalization_ranges))):
                l_min, l_max = self.normalization_ranges[i]
                raster[i] = np.clip(raster[i], l_min, l_max)
                raster[i] = (raster[i] - l_min) / (l_max - l_min)
        
        raster = raster / 255.0  # Normalize to [0, 1]
        return raster, tile


def custom_collate_fn(batch):
    """Custom collate function for handling batch processing."""
    batch = [item for item in batch if item is not None]  # Filter out None values
    if len(batch) == 0:  # If the entire batch is invalid, return empty placeholders
        return None, None
    images, tiles = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, tiles

def read_data(geojson_path, vrt_file):
    dataset = GeoTileDataset(geojson_path, vrt_file, normalization_ranges =[(350, 1600), (500, 1600), (600, 1800)]) ## ranges of Sentinel2 images
    dataloader = DataLoader(dataset, batch_size=16, num_workers=4, collate_fn=custom_collate_fn)
    
    return dataset, dataloader

def plot_examples(dataset, num_examples=5):
    """
    Plot examples from the dataset.

    Parameters:
        dataset (GeoTileDataset): The dataset to visualize.
        num_examples (int): Number of examples to plot.
    """
    for i in range(min(num_examples, len(dataset))):
        raster, tile = dataset[i]
        raster = raster * 255.0
        raster = raster.numpy().transpose(1, 2, 0)  # Convert back to HWC format for plotting
        centroid = tile.geometry.centroid

        plt.figure(figsize=(8, 8))
        plt.imshow(raster)
        plt.title(f"Tile Centroid: ({centroid.x:.2f}, {centroid.y:.2f})")
        plt.axis("off")
        plt.show()


def load_model(model_weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.swin_transformer.swin_v2_b().to(device)
    full_state_dict = torch.load(model_weights_path, map_location=device, weights_only=True)

    swin_prefix = 'backbone.backbone.'
    fpn_prefix = 'intermediates.0.fpn.'  # FPN

    swin_state_dict = {k[len(swin_prefix):]: v for k, v in full_state_dict.items() if k.startswith(swin_prefix)}
    model.load_state_dict(swin_state_dict)

    fpn_state_dict = {k[len(fpn_prefix):]: v for k, v in full_state_dict.items() if k.startswith(fpn_prefix)}
    fpn = FeaturePyramidNetwork([128, 256, 512, 1024], out_channels=128).to(device)
    fpn.load_state_dict(fpn_state_dict)

    return model, fpn


def extract_embeddings(dataloader, model, fpn, device):
    #model.eval() # not neccessary
    embeddings = []
    tiles = []
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:  # Skip empty batches
                continue
            images, meta = batch
            images = images.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = []
                x = images
                for layer in model.features:
                    x = layer(x)
                    outputs.append(x.permute(0, 3, 1, 2))
                map1, map2, map3, map4 = outputs[-7], outputs[-5], outputs[-3], outputs[-1]

                # Process feature maps with FPN
                feature_maps_raw = [map1, map2, map3, map4]
                inp = collections.OrderedDict([('feat{}'.format(i), el) for i, el in enumerate(feature_maps_raw)])
                fpn_output = fpn(inp)
                fpn_output = list(fpn_output.values())

                avgpool = torch.nn.AdaptiveAvgPool2d(1)
                features = avgpool(fpn_output[-1])[:, :, 0, 0]  # Global avg pooling

            embeddings.append(features.cpu().numpy())
            tiles.extend(meta)

    return np.vstack(embeddings), tiles


def extract_feature_vectors(batch, model, fpn, device):
    images, paths = batch
    images = images.to(device)

    # Run the images through the model and extract features
    with autocast(), torch.no_grad():
        outputs = []
        x = images
        for layer in model.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2))
        map1, map2, map3, map4 = outputs[-7], outputs[-5], outputs[-3], outputs[-1]

        # Process feature maps with FPN
        feature_maps_raw = [map1, map2, map3, map4]
        inp = collections.OrderedDict([('feat{}'.format(i), el) for i, el in enumerate(feature_maps_raw)])
        fpn_output = fpn(inp)
        fpn_output = list(fpn_output.values())

        avgpool = torch.nn.AdaptiveAvgPool2d(1)
        features = avgpool(fpn_output[-1])[:, :, 0, 0]  # Global avg pooling

    return features.cpu().numpy(), paths



def classify_tiles(embeddings, tiles, classifier, transformer, h3_resolution):
    lon_lat = []
    features_with_h3 = []

    for tile in tiles:
        centroid = tile.geometry.centroid
        lon, lat = transformer.transform(centroid.x, centroid.y)

        #h3_resolution = 6 # not fixed anymore added to function
        h3_index = h3.latlng_to_cell(lat, lon, h3_resolution)
        #lat_h5, lon_h5 = h3.h3shape_to_geo(h3_index) #previous version
        lat_h5, lon_h5 = h3.cell_to_latlng(h3_index)

        lon_lat.append((lon, lat))
        features_with_h3.append([lon_h5, lat_h5])

    # Combine embeddings with H3 lat/lon
    combined_features = np.hstack([embeddings, np.array(features_with_h3)])

    predictions = classifier.predict(combined_features)
    probas = classifier.predict_proba(combined_features)

    results = []
    for idx, (lon, lat) in enumerate(lon_lat):
        tile_result = {
            "longitude": lon,
            "latitude": lat,
            "prediction": predictions[idx],
            "probabilities": probas[idx].tolist(),
            "lon_h3": features_with_h3[idx][0],
            "lat_h3": features_with_h3[idx][1],
        }
        results.append(tile_result)

    return pd.DataFrame(results)


def save_to_geoparquet(results, geo_path, output_path):
    gdf = gpd.GeoDataFrame(
        results,
        geometry=gpd.points_from_xy(results["longitude"], results["latitude"]),
        crs="EPSG:4326",
    )
    gdf_ = gdf.to_crs(27700)
    aoi = gpd.read_file(geo_path)

    gdf_fin = gpd.sjoin(aoi, gdf_, how="inner", predicate="contains")

    gdf_fin = gdf_fin[['id', 'prediction', 'probabilities', 'lon_h3', 'lat_h3', 'geometry']] 
    gdf_fin.to_parquet(output_path, engine="pyarrow", index=False)
    

def spatial_sig_prediction(geo_path, vrt_file, model_weights, xgb_weights, output_path, h3_resolution):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, fpn = load_model(model_weights)
    
    dataset = GeoTileDataset(geo_path, vrt_file, normalization_ranges =[(350, 1600), (500, 1600), (600, 1800)])
    dataloader = DataLoader(dataset, batch_size=16, num_workers=4, collate_fn=custom_collate_fn)
    
    embeddings, tiles = extract_embeddings(dataloader, model, fpn, device)
    pd.DataFrame(embeddings).to_csv('/bask/homes/f/fedu7800/vjgo8416-demoland/satellite_demoland/data/london_emb.csv')

    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.load_model(xgb_weights)

    results = classify_tiles(embeddings, tiles, xgb_classifier, transformer, h3_resolution=h3_resolution) #ss_gdf
    save_to_geoparquet(
        results, geo_path, output_path)


