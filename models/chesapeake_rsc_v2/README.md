# ChesapeakeRSC v2 roads

This RGB model proposes road centerlines for mapper review, including roads hidden by tree canopy. The U-Net uses a ResNet-18 encoder and predicts background and road logits. It was pretrained for road segmentation on 0.6 m NAIP imagery in the U.S. Mid-Atlantic.

The [model weights](https://huggingface.co/isaaccorley/chesapeakersc-v2) are MIT-licensed. The [CC0-1.0 dataset](https://huggingface.co/datasets/isaaccorley/chesapeakersc-v2) includes the chip schema, manifests, and held-out county splits. Its raster labels distinguish background, visible roads, roads under canopy, and ignored pixels. On 11,345 held-out chips, the RGB checkpoint achieved 79.0% road pixel F1 and 53.5% canopy recall. On 6,385 Delaware chips it achieved 82.6% F1 and 66.1% canopy recall. These metrics describe the raster model before centerline extraction.

## Inference and fine-tuning

Use the [fAIr model contribution guide](https://hotosm.github.io/fAIr-models/contributing/model/) to build the image and run the local example and serving test. CPU ONNX Runtime is sufficient for inference. The supplied training image uses CPU PyTorch. A CUDA-enabled PyTorch environment can accelerate larger fine-tuning jobs. Memory use grows with chip area and training batch size. Start with a batch size of one or two on new imagery.

The pipeline reads georeferenced uint8 RGB GeoTIFFs, resamples them onto a common grid at approximately 0.6 m and divides by 255. It runs the U-Net on windows up to 512 × 512 pixels with 50% overlap and Hann-averages their road probabilities. By default it traces ridges in the blended probabilities, with a 0.08 floor on the smoothed probabilities. The mask fallback thresholds the probabilities at 0.25. Both methods trace across tile seams. Each GeoJSON feature is a WGS84 LineString with integer class 1 and label road. Send adjacent chips from a single north-up imagery grid in the same request. Split large requests into contiguous areas of at most 16 million pixels at 0.6 m GSD.

Fine-tuning starts from the published RGB PyTorch weights and expects same-zoom OAM-named chips and WGS84 road-only GeoJSON labels. Road polygons rasterize directly. Centerlines are buffered to the configurable full road width, 4 m by default. Unlabeled valid pixels count as background, so use chips with complete road labels. Use at least two spatial blocks so training and validation can hold out whole blocks. The validation step reports pixel F1 and IoU, and the training pipeline exports a self-contained ONNX file.

## Limitations and responsible use

Performance varies with imagery source, resolution, seasons, and local road materials. Resampling cannot recover detail absent from the input imagery. Occlusion can lead to missed roads, false connections, or disconnected centerlines, especially near junctions and tile edges. Review predictions against local imagery and mapping knowledge before adding them to OpenStreetMap or using them for routing or access planning.

## Licensing and citation

The weights and Chesapeake-specific model code are offered under MIT. The enclosing fAIr project is AGPL-3.0-only. Direct model dependencies include PyTorch and torchvision (BSD-3-Clause), segmentation-models-pytorch (MIT), ONNX and ONNX Runtime (Apache-2.0 and MIT respectively), NumPy, SciPy, and scikit-image (BSD-3-Clause), Rasterio (BSD-3-Clause), Affine (BSD-3-Clause), Shapely (BSD-3-Clause), pyproj (MIT), ZenML (Apache-2.0), MLflow (Apache-2.0), and the fAIr runtime (AGPL-3.0-only), which uses geomltoolkits (MIT) to download imagery. See the Dockerfile for the runtime environment.

Caleb Robinson, Isaac Corley, Anthony Ortiz, Rahul Dodhia, Juan M. Lavista Ferres, and Peyman Najafirad. [Seeing the Roads Through the Trees](https://arxiv.org/abs/2401.06762), 2024.
