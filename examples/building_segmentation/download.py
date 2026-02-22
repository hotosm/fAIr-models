"""Download OAM imagery and OSM building labels for Banepa Municipality, Nepal.

Edit constants at the top to adapt to a different area or OAM image.
"""

from __future__ import annotations

import os

from torch.utils.data import DataLoader
from torchgeo.datasets import OpenAerialMap, OpenStreetMap, stack_samples
from torchgeo.samplers import RandomGeoSampler

# Smaller bbox (~400m x 400m, central Banepa) for fast test runs; extend for production.
BBOX = (85.519, 27.630, 85.523, 27.634)  # (minx, miny, maxx, maxy)
ZOOM = 19
OAM_IMAGE_ID = "62d86c65d8499800053796c4"
CHIP_SIZE = 512
WORK_DIR = "data/banepa_test"
BUILDING_CLASSES = [{"name": "building", "selector": [{"building": "*"}]}]
SAMPLER_LENGTH = 10

oam_dir = os.path.join(WORK_DIR, "oam")
osm_dir = os.path.join(WORK_DIR, "osm")
os.makedirs(oam_dir, exist_ok=True)
os.makedirs(osm_dir, exist_ok=True)

oam = OpenAerialMap(
    paths=oam_dir,
    bbox=BBOX,
    zoom=ZOOM,
    image_id=OAM_IMAGE_ID,
    download=True,
    tile_size=CHIP_SIZE,
)
b = oam.bounds
osm = OpenStreetMap(
    bbox=(b[0].start, b[1].start, b[0].stop, b[1].stop),
    classes=BUILDING_CLASSES,
    paths=osm_dir,
    download=True,
)

dataset = oam & osm
sampler = RandomGeoSampler(dataset, size=CHIP_SIZE, length=SAMPLER_LENGTH)
loader = DataLoader(dataset, sampler=sampler, batch_size=1, collate_fn=stack_samples)

batch = next(iter(loader))
print(f"OAM dir:   {oam_dir}")
print(f"OSM dir:   {osm_dir}")
print(f"Image shape:  {tuple(batch['image'].shape)}")
print(f"Mask shape:   {tuple(batch['mask'].shape)}")
print(f"Dataset bounds: {dataset.bounds}")
