"""Download OAM imagery and OSM building labels for Banepa, Nepal."""

from __future__ import annotations

import os

from torch.utils.data import DataLoader
from torchgeo.datasets import OpenAerialMap, OpenStreetMap, stack_samples
from torchgeo.samplers import RandomGeoSampler

ZOOM = 19
OAM_IMAGE_ID = "62d86c65d8499800053796c4"
CHIP_SIZE = 512
BUILDING_CLASSES = [{"name": "building", "selector": [{"building": "*"}]}]

# Train area: central Banepa (~400m x 400m)
TRAIN_BBOX = (85.519, 27.630, 85.523, 27.634)
TRAIN_DIR = "data/sample/train"

# Predict area: shifted east (~200m x 200m), no overlap with train
PREDICT_BBOX = (85.524, 27.631, 85.526, 27.633)
PREDICT_DIR = "data/sample/predict"

train_oam_dir = os.path.join(TRAIN_DIR, "oam")
train_osm_dir = os.path.join(TRAIN_DIR, "osm")
os.makedirs(train_oam_dir, exist_ok=True)
os.makedirs(train_osm_dir, exist_ok=True)

print("Downloading training OAM imagery...")
train_oam = OpenAerialMap(
    paths=train_oam_dir,
    bbox=TRAIN_BBOX,
    zoom=ZOOM,
    image_id=OAM_IMAGE_ID,
    download=True,
    tile_size=CHIP_SIZE,
)
b = train_oam.bounds
print("Downloading training OSM labels...")
train_osm = OpenStreetMap(
    bbox=(b[0].start, b[1].start, b[0].stop, b[1].stop),
    classes=BUILDING_CLASSES,
    paths=train_osm_dir,
    download=True,
)

dataset = train_oam & train_osm
sampler = RandomGeoSampler(dataset, size=CHIP_SIZE, length=5)
loader = DataLoader(dataset, sampler=sampler, batch_size=1, collate_fn=stack_samples)
batch = next(iter(loader))
print(f"Train OAM:   {train_oam_dir} ({len(os.listdir(train_oam_dir))} files)")
print(f"Train OSM:   {train_osm_dir}")
print(f"Image shape: {tuple(batch['image'].shape)}")
print(f"Mask shape:  {tuple(batch['mask'].shape)}")

predict_oam_dir = os.path.join(PREDICT_DIR, "oam")
os.makedirs(predict_oam_dir, exist_ok=True)

print("\nDownloading prediction OAM imagery (no labels)...")
predict_oam = OpenAerialMap(
    paths=predict_oam_dir,
    bbox=PREDICT_BBOX,
    zoom=ZOOM,
    image_id=OAM_IMAGE_ID,
    download=True,
    tile_size=CHIP_SIZE,
)
print(f"Predict OAM: {predict_oam_dir} ({len(os.listdir(predict_oam_dir))} files)")
print("\nDone.")
