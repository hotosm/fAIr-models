# Line-by-Line Explanation of `inside_container_smoke_test.py`

A beginner-friendly walkthrough of the RAMP smoke test script.

---

## What is `# noqa: F401`?

**F401** is a **Flake8** (Python linter) rule code. It means:

- **F** = Pyflakes (the part of Flake8 that checks for logic/import issues)
- **401** = "Module imported but unused"

So **F401** = "You imported a module but never use any name from it."

When we write:

```python
import tensorflow as tf  # noqa: F401
```

we are:

1. **Importing** the `tensorflow` package so Python loads it (and we check it’s installed).
2. **Not using** `tf` later in a way the linter sees (we do use `tf.__version__` and `tf.keras` elsewhere, but on *this* line we only care that the import works).
3. **Silencing the linter** with `# noqa: F401` so it doesn’t report "imported but unused" for this line.

So: **F401 = "imported but unused"**, and **noqa: F401** means "don’t warn about F401 on this line."

---

## Top of the file (docstring and imports)

```python
"""End-to-end smoke tests for models/ramp Docker runtime.
...
"""
```

- **Triple-quoted string** at the top of a file is the **docstring** for the module. It describes what the script does. Tools and `help()` can show it.

```python
from __future__ import annotations
```

- Makes type hints (like `str`, `Path`) be treated as strings by the interpreter. Helps with forward references and cleaner type hints.

```python
import argparse
import os
import shutil
from pathlib import Path
```

- **argparse**: read command-line arguments (e.g. `--dataset-root`, `--epochs`).
- **os**: environment variables, path checks (e.g. `os.environ`, `os.path`).
- **shutil**: copy/delete trees (e.g. `shutil.rmtree`, `shutil.copy`).
- **pathlib.Path**: object-oriented paths (`Path("a") / "b"` → `a/b`), `.is_dir()`, `.glob()`, etc.

---

## Helper functions

```python
def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
```

- **def**: define a function.
- **condition: bool**: argument must be a boolean (True/False).
- **message: str**: argument must be a string.
- **-> None**: function returns nothing.
- If `condition` is False, we raise `RuntimeError(message)` so the test fails with a clear message. The leading `_` means "internal/private" by convention.

```python
def _stage(name: str) -> None:
    print(f"\n=== {name} ===")
```

- Prints a section header like `\n=== Test 1: Critical imports ===`. The **f-string** `f"..."` lets you embed `{name}` in the string.

```python
def _count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern)))
```

- **path.glob(pattern)**: finds all files under `path` matching `pattern` (e.g. `"*.png"`). Returns an iterator.
- **list(...)**: turn that iterator into a list.
- **len(...)**: number of items. So this returns how many files match the pattern.

---

## Parsing command-line arguments

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAMP container smoke tests.")
```

- **ArgumentParser**: object that will define and parse CLI arguments.
- **argparse.Namespace**: type of the object you get back (e.g. `args.dataset_root`).

```python
    parser.add_argument(
        "--dataset-root",
        default="/workspace/data/sample",
        help="...",
    )
```

- Adds an option `--dataset-root`. If the user doesn’t pass it, `args.dataset_root` is `"/workspace/data/sample"`.
- **help**: text shown when you run `python script.py --help`.

```python
    parser.add_argument("--epochs", type=int, default=2)
```

- **type=int**: value is converted to an integer.
- Same idea for **--batch-size** and **--backbone** (string).

```python
    return parser.parse_args()
```

- Reads `sys.argv` (the actual command line), fills in the values, and returns an object with attributes like `args.dataset_root`, `args.epochs`, etc.

---

## Preparing data/sample layout

```python
def _prepare_data_sample_layout(dataset_root: Path) -> Path:
```

- Takes a path to the dataset root, returns the path to the "work" root (where we’ll put prepared input).

```python
    oam_dir = dataset_root / "train" / "oam"
    osm_dir = dataset_root / "train" / "osm"
```

- **Path / "train" / "oam"**: builds path like `dataset_root/train/oam`. `/` on `Path` joins path parts.

```python
    _assert(oam_dir.is_dir(), f"train/oam not found under {dataset_root}")
```

- Fails fast if `train/oam` (or `train/osm`) doesn’t exist.

```python
    work_root = dataset_root / "ramp_work"
    input_dir = work_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
```

- **mkdir(parents=True, exist_ok=True)**: create the directory and any missing parents; don’t error if it already exists.

```python
    import numpy as np
    import rasterio
    from PIL import Image
```

- Imports used only in this function. **numpy**: arrays/math. **rasterio**: read/write rasters (e.g. GeoTIFF). **PIL.Image**: image I/O (e.g. save PNG).

```python
    tif_files = sorted(oam_dir.glob("OAM-*.tif"))
```

- **glob("OAM-*.tif")**: all files whose names match that pattern. **sorted()**: stable order for reproducibility.

```python
    for tif_path in tif_files:
        png_path = input_dir / (tif_path.stem + ".png")
```

- **tif_path.stem**: filename without extension (e.g. `OAM-386695-220244-19`). So we get a path like `input_dir/OAM-386695-220244-19.png`.

```python
        with rasterio.open(tif_path) as src:
            data = src.read()
```

- **with ... as src**: open the file and automatically close it when the block ends. **src.read()**: read all bands as a 3D array (bands, height, width).

```python
            if data.shape[0] >= 3:
                rgb = np.transpose(data[:3], (1, 2, 0))
```

- **data.shape[0]**: number of bands. We need at least 3 (R, G, B).
- **data[:3]**: first 3 bands. **np.transpose(..., (1, 2, 0))**: change layout from (bands, H, W) to (H, W, bands) for image-style arrays.

```python
                if rgb.max() <= 1.0:
                    rgb = (rgb * 255).astype(np.uint8)
                Image.fromarray(rgb).save(png_path)
```

- If values are in 0–1, scale to 0–255 and convert to 8-bit integer. **Image.fromarray(rgb).save(png_path)**: write a PNG file.

```python
    geojson_files = sorted(osm_dir.glob("*.geojson"))
    ...
    gdfs = [gpd.read_file(p) for p in geojson_files]
```

- **gpd.read_file(p)**: read a GeoJSON into a GeoDataFrame. The list comprehension builds a list of one GeoDataFrame per file.

```python
    if len(gdfs) == 1:
        merged = gdfs[0]
    else:
        import pandas as pd
        crs = gdfs[0].crs or "EPSG:4326"
        merged = gpd.GeoDataFrame(
            pd.concat([g.to_crs(crs) for g in gdfs], ignore_index=True),
            crs=crs,
        )
```

- One file → use it as-is. Multiple → convert all to the same CRS (**crs**), concatenate with **pd.concat**, wrap in **GeoDataFrame**, and assign **crs**.

```python
    labels_path = input_dir / "labels.geojson"
    merged.to_file(labels_path, driver="GeoJSON")
```

- Write the (possibly merged) labels to **input/labels.geojson** so the rest of the pipeline sees a single labels file.

```python
    return work_root
```

- Return the path to the directory that now contains **input/** with PNGs and **labels.geojson**.

---

## main() — entry point

```python
def main() -> None:
    args = parse_args()
```

- Parse CLI into **args**.

```python
    os.environ.setdefault("RAMP_HOME", "/workspace")
```

- Set **RAMP_HOME** only if it’s not already set. Some RAMP code uses this.

```python
    dataset_root = Path(args.dataset_root).resolve()
```

- **Path(...)**: turn the string into a Path. **.resolve()**: make it absolute and normalize (e.g. resolve `..`).

```python
    if (dataset_root / "train" / "oam").is_dir() and (dataset_root / "train" / "osm").is_dir():
        dataset_root = _prepare_data_sample_layout(dataset_root)
```

- If we see the **data/sample** layout (train/oam and train/osm), we prepare it and then **dataset_root** becomes the work root (**ramp_work**).

```python
    input_dir = dataset_root / "input"
    preprocessed_dir = dataset_root / "preprocessed_test"
    chips_dir = preprocessed_dir / "chips"
    ...
```

- Define all the paths we’ll use for inputs and outputs (chips, masks, val split, checkpoints, prediction output, vectors). No files are created yet; these are just path variables.

---

## Test 1: Critical imports

```python
    import segmentation_models as sm  # noqa: F401
    import tensorflow as tf  # noqa: F401
    from osgeo import gdal  # noqa: F401
    import ramp  # noqa: F401
    import hot_fair_utilities  # noqa: F401
```

- We import these **to check they are installed and loadable**. If any import fails, the script crashes and the test fails. We use **noqa: F401** because the linter would otherwise say "imported but unused" on some of these lines (we do use `sm`, `tf`, etc. later; the comment keeps the linter happy where it still complains).

```python
    sm.set_framework("tf.keras")
```

- Tell **segmentation_models** to use TensorFlow/Keras as the backend.

```python
    print(f"PASS: tensorflow {tf.__version__} ...")
```

- **tf.__version__**: the TensorFlow version string. Printing it confirms the import worked and helps with debugging.

---

## Test 2: Dataset layout

```python
    _assert(dataset_root.is_dir(), ...)
    _assert(input_dir.is_dir(), ...)
    _assert((input_dir / "labels.geojson").is_file(), ...)
```

- Ensure the expected folders and **labels.geojson** exist; otherwise raise with a clear message.

```python
    n_png = _count_files(input_dir, "*.png")
    _assert(n_png > 0, ...)
```

- Count PNGs in **input/** and require at least one.

---

## Test 3: Preprocessing

```python
    shutil.rmtree(preprocessed_dir, ignore_errors=True)
```

- Remove any previous preprocessed output so we start clean. **ignore_errors=True**: don’t fail if the directory doesn’t exist.

```python
    from hot_fair_utilities import preprocess as _preprocess
    _preprocess(
        input_path=str(input_dir),
        output_path=str(preprocessed_dir),
        ...
    )
```

- **import ... as _preprocess**: we only call the function; the leading `_` suggests "used locally."
- **str(...)**: some APIs expect strings, not **Path**; we convert so it works everywhere.

```python
    n_chips = _count_files(chips_dir, "*.tif")
    n_masks = _count_files(masks_dir, "*.mask.tif")
    _assert(n_chips == n_masks, ...)
```

- After preprocessing we expect one chip and one mask per image; we check that the counts match.

---

## Test 4: Train/val split

```python
    chip_files = sorted(chips_dir.glob("*.tif"))
    n_val = max(1, int(len(chip_files) * 0.2))
```

- **n_val**: 20% of chips for validation, but at least 1.

```python
    random.shuffle(chip_files)
    val_chip_files = chip_files[:n_val]
```

- Shuffle and take the first **n_val** as validation; the rest stay as training.

```python
    for chip_path in val_chip_files:
        mask_path = masks_dir / (chip_path.stem + ".mask.tif")
        if mask_path.is_file():
            shutil.copy(str(chip_path), val_chips_dir / chip_path.name)
            shutil.copy(str(mask_path), val_masks_dir / mask_path.name)
            moved += 1
```

- For each validation chip, find the matching mask by name (e.g. `chip.tif` → `chip.mask.tif`). If it exists, copy both chip and mask into the val directories and count.

---

## Test 5: Training

- **cfg**: a big dictionary that RAMP uses for training (epochs, batch size, loss, optimizer, callbacks, etc.). This is the "config" for the run.
- **loss_fn**, **optimizer**, **acc_metric**: built from RAMP’s constructors using **cfg**.
- **sm.Unet(...)**: build the U-Net model. **encoder_weights=None** avoids downloading pretrained weights (which can 404 in CI).
- **the_model.compile(...)**: attach optimizer, loss, and metric for training.
- **training_batches_from_gtiff_dirs** / **test_batches_from_gtiff_dirs**: RAMP functions that yield batches from the chip and mask directories.
- **the_model.fit(...)**: run training for **args.epochs** (e.g. 2) with the given batches and callbacks.
- **the_model.save(model_save_path)**: save the trained model so we can load it in the next stage.

---

## Test 6: Inference

```python
    inference_model = tf.keras.models.load_model(model_save_path, compile=False)
```

- Load the SavedModel we just saved. **compile=False**: we don’t need the training graph, only forward pass.

```python
    for chip_file in chip_files[:3]:
```

- Run prediction on only the first 3 chips to keep the smoke test fast.

```python
        with rio.open(chip_file) as src:
            dst_profile = src.profile.copy()
            ...
            img = to_channels_last(src.read()).astype("float32")
```

- **src.profile**: metadata (size, dtype, etc.) we reuse for the output. **to_channels_last**: convert from (C, H, W) to (H, W, C) if needed for the model.

```python
            predicted = get_mask_from_prediction(inference_model.predict(np.expand_dims(img, 0)))
```

- **np.expand_dims(img, 0)**: add a batch dimension (1, H, W, C). **predict(...)**: run the model. **get_mask_from_prediction**: turn model output into a class mask. Then we write the result as **.pred.tif**.

---

## Test 7: Polygonization

```python
        ref_ds = gdal.Open(str(pred_tif))
        multimask = gdal_get_mask_tensor(str(pred_tif))
        bin_mask = binary_mask_from_multichannel_mask(multimask)
        binary_mask_to_geojson(bin_mask, ref_ds, json_path)
```

- **gdal.Open**: open the predicted GeoTIFF (for georeference info).
- **gdal_get_mask_tensor**: read the mask array.
- **binary_mask_from_multichannel_mask**: reduce multi-class mask to a single binary mask (e.g. building vs non-building).
- **binary_mask_to_geojson**: raster-to-vector (polygonize) and write building footprints to GeoJSON.

---

## End of script

```python
if __name__ == "__main__":
    main()
```

- **__name__**: special variable. When you run this file as a script, **__name__** is **"__main__"**. When the file is imported, **__name__** is the module name.
- So **main()** runs only when you execute the script (e.g. `python inside_container_smoke_test.py`), not when another file does `import inside_container_smoke_test`.

---

## Quick reference

| Concept | Meaning |
|--------|--------|
| **F401** | Flake8: "module imported but unused" |
| **noqa: F401** | "Don’t report F401 on this line" |
| **Path / "a" / "b"** | Path joining: `path/a/b` |
| **path.glob("*.png")** | All files matching pattern under **path** |
| **with open(...) as f:** | Use **f** and close the resource when the block ends |
| **f"text {var}"** | f-string: insert **var** into the string |
| **-> None** | This function returns nothing |
| **type=int** | Convert CLI argument to int |
