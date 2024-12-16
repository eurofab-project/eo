import os
import glob
import torch
import torchvision
import xgboost as xgb
import pandas as pd
import numpy as np
import tifffile
import collections
import argparse
from pathlib import Path
from torchvision.ops import FeaturePyramidNetwork
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast

import dask.dataframe as dd

import tools_sat_ss as tools

#########################################################
# Step 1: Chip Extraction (from VRT to TIF Chips)
#########################################################


def calculate_centroids_partition(df):
    centroids = df.geometry.centroid

    return pd.DataFrame({
        "centroid_x": centroids.x,
        "centroid_y": centroids.y
    }, index=df.index)

def sample_partition(df):
    return df.groupby('type_clean').apply(
        lambda x: x.sample(n=min(50000, len(x)), random_state=42)
    )

def create_chips_from_vrt(grid_25_file, signatures_file, specs, npartitions=16):
    """
    Creates 25x25 pixel chips from a large VRT mosaic and saves them as individual .tif files.
    
    geo_df: GeoDataFrame containing geometry and metadata for each chip location
    specs: dict with fields like:
        {
            'chip_size': 25,
            'bands': [1, 2, 3],
            'mosaic_p': '/path/to/GHS-composite-S2.vrt',
            'folder': '/path/to/output/chips/',
            'normalize': True,
            'normalization_ranges': [(350,1600),(500,1600),(600,1800)]
        }
    npartitions: number of Dask partitions
    """

    # Load grid and spatial signatures
    grid_25 = gpd.read_file(grid_25_file)
    ss_gdf = gpd.read_file(signatures_file, layer='spatial_signatures_GB_clean')
    ss_gdf = ss_gdf[ss_gdf['type'] != 'outlier']

    # Define urbanity classes and clean 'type' column
    urbanity_classes = ['Local urbanity', 'Regional urbanity', 'Metropolitan urbanity', 'Concentrated urbanity', 'Hyper concentrated urbanity']
    ss_gdf['type_clean'] = ss_gdf['type'].apply(lambda x: 'Urbanity' if x in urbanity_classes else x)
    le = LabelEncoder()
    ss_gdf['type_clean_encode'] = le.fit_transform(ss_gdf['type_clean'])

    overlap_25 = dgpd.sjoin(grid_25, ss_gdf, how='inner', predicate='within').compute()

    sel = grid_25[grid_25['id'].isin(overlap_25['id_left'])]
    sel_dask = dgpd.from_geopandas(sel, npartitions=4)  


    # Use `meta` for output format specification
    meta = pd.DataFrame({"centroid_x": pd.Series(dtype="float64"), "centroid_y": pd.Series(dtype="float64")})

    # Apply the function to each partition
    centroids = sel_dask.map_partitions(calculate_centroids_partition, meta=meta).compute()


    # Convert centroids to a Dask DataFrame
    centroids_dd = dd.from_pandas(pd.DataFrame(centroids, columns=["centroid_x", "centroid_y"]), npartitions=sel_dask.npartitions)

    # Assign each column to `sel` as a Dask Series
    sel["X"] = centroids_dd["centroid_x"]
    sel["Y"] = centroids_dd["centroid_y"]

    # Perform the second spatial join
    sel_25 = dgpd.sjoin(sel, ss_gdf, how="inner", predicate="within").compute()
    sel_25 = sel_25[["X", "Y", "type_clean", "geometry"]]

    sampled_df = sel_25_repartitioned.map_partitions(sample_partition).compute()
    sampled_df['type_clean'] = sampled_df['type_clean'].replace('Warehouse/Park land', 'Warehouse')

    tools.spilled_bag_of_chips(sampled_df, specs, npartitions=npartitions)

#########################################################
# Step 2: Embeddings Extraction
#########################################################

def load_model(model_weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.swin_transformer.swin_v2_b().to(device)

    full_state_dict = torch.load(model_weights_path, map_location=device)

    swin_prefix = 'backbone.backbone.'
    fpn_prefix = 'intermediates.0.fpn.'

    swin_state_dict = {k[len(swin_prefix):]: v for k, v in full_state_dict.items() if k.startswith(swin_prefix)}
    model.load_state_dict(swin_state_dict)

    fpn_state_dict = {k[len(fpn_prefix):]: v for k, v in full_state_dict.items() if k.startswith(fpn_prefix)}
    fpn = FeaturePyramidNetwork([128, 256, 512, 1024], out_channels=128).to(device)
    fpn.load_state_dict(fpn_state_dict)

    return model, fpn

class TIFDataset(Dataset):
    def __init__(self, image_paths, normalize=False):
        self.image_paths = image_paths
        self.normalize = normalize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        im = tifffile.imread(image_path)
        im = im.astype(float)

        # Move to [C, H, W]
        im = im.transpose(2, 0, 1) 
        return torch.from_numpy(im).float(), image_path

def extract_feature_vectors(batch, model, fpn, device):
    images, paths = batch
    images = images.to(device)

    with autocast(), torch.no_grad():
        outputs = []
        x = images
        for layer in model.features:
            x = layer(x)
            outputs.append(x.permute(0, 3, 1, 2))
        map1, map2, map3, map4 = outputs[-7], outputs[-5], outputs[-3], outputs[-1]

        inp = collections.OrderedDict([('feat{}'.format(i), el) for i, el in enumerate([map1, map2, map3, map4])])
        fpn_output = fpn(inp)
        fpn_output = list(fpn_output.values())

        avgpool = torch.nn.AdaptiveAvgPool2d(1)
        features = avgpool(fpn_output[-1])[:, :, 0, 0]

    return features.cpu().numpy(), paths

def process_images_in_batches(folder_path, model, fpn, batch_size=16, normalize=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    fpn.to(device)

    image_paths = sorted(glob.glob(os.path.join(folder_path, "*/*.tif")))

    dataset = TIFDataset(image_paths, normalize=normalize)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    all_features = []
    all_paths = []

    print("Processing images in batches...")
    for batch in dataloader:
        features, paths = extract_feature_vectors(batch, model, fpn, device)
        all_features.append(features)
        all_paths.extend(paths)

    all_features = np.vstack(all_features)
    return all_features, all_paths

def save_features_to_parquet(features, paths, output_file):
    print("Saving embeddings to parquet...")
    df = pd.DataFrame(features)
    df['file_path'] = paths
    df.to_parquet(output_file, index=False)
    print(f"Features saved to {output_file}")

def run_embeddings_model(input_folder, output_file, model_weights_path, batch_size=16, normalize=False):
    model, fpn = load_model(model_weights_path)
    features, paths = process_images_in_batches(input_folder, model, fpn, batch_size=batch_size, normalize=normalize)
    save_features_to_parquet(features, paths, output_file)

#########################################################
# Step 3: Convert Embeddings for ML Model Input
#########################################################

def load_embeddings_from_parquet(embeddings_file):
    df = pd.read_parquet(embeddings_file)
    # Separate features from paths
    paths = df['file_path']
    X = df.drop('file_path', axis=1).values
    return X, paths

# Optionally save as .h5 if needed
def save_embeddings_h5(X, h5_file):
    import h5py
    with h5py.File(h5_file, 'w') as f:
        f.create_dataset('embeddings', data=X)

def load_embeddings_h5(h5_file):
    import h5py
    with h5py.File(h5_file, 'r') as f:
        X = f['embeddings'][:]
    return X

#########################################################
# Step 4: Run ML Model (XGBoost)
#########################################################

def run_ML_model(XGBoost_model_file, X):
    xgb_model = xgb.XGBClassifier(
        objective='multi:softmax',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42,
        enable_categorical=False,
    )
    xgb_model.load_model(XGBoost_model_file)

    predicted_values = xgb_model.predict(X)
    predicted_probs = xgb_model.predict_proba(X)
    return predicted_values, predicted_probs

#########################################################
# Step 5: Uncertainty Calculation
#########################################################

def calculate_uncertainty(predicted_probs, method='entropy'):
    """
    Calculate uncertainty from predicted probabilities.
    method='entropy': uses Shannon entropy
    """
    if method == 'entropy':
        # Entropy = -sum(p * log(p))
        entropy = -(predicted_probs * np.log(predicted_probs+1e-10)).sum(axis=1)
        return entropy
    else:
        # Add other methods if desired, e.g., top-2 margin
        # margin = difference between top 2 predicted classes
        # ...
        raise NotImplementedError("Only 'entropy' method implemented")

#########################################################
# Final Step: Save Predictions and Uncertainty
#########################################################

def save_predictions_to_parquet(paths, predicted_values, predicted_probs, uncertainty, output_file):
    df = pd.DataFrame({
        'file_path': paths,
        'prediction': predicted_values,
        'uncertainty': uncertainty
    })
    # If you want, you can also save predicted_probs as multiple columns:
    for i in range(predicted_probs.shape[1]):
        df[f'prob_class_{i}'] = predicted_probs[:, i]

    df.to_parquet(output_file, index=False)
    print(f"Predictions saved to {output_file}")

#########################################################
# Example Pipeline Usage
#########################################################

if __name__ == "__main__":
    # Example usage, assuming necessary inputs and paths
    # 1. Create chips from VRT
    # geo_df = ... load your geodataframe here ...
    # specs = { ... } # specify your parameters
    # create_chips_from_vrt(geo_df, specs)

    # 2. Extract embeddings
    # run_embeddings_model(input_folder="/path/to/chips",
    #                      output_file="embeddings.parquet",
    #                      model_weights_path="/path/to/model_weights.pt",
    #                      batch_size=16,
    #                      normalize=True)

    # 3. Load embeddings
    # X, paths = load_embeddings_from_parquet("embeddings.parquet")

    # (Optional) convert to h5
    # save_embeddings_h5(X, "embeddings.h5")
    # X = load_embeddings_h5("embeddings.h5")

    # 4. Run ML model
    # predicted_values, predicted_probs = run_ML_model("model.xgb", X)

    # 5. Calculate uncertainty
    # uncertainty = calculate_uncertainty(predicted_probs, method='entropy')

    # 6. Save predictions
    # save_predictions_to_parquet(paths, predicted_values, predicted_probs, uncertainty, "predictions.parquet")

    pass
