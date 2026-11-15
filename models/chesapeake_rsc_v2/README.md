# ChesapeakeRSC v2 roads

ChesapeakeRSC v2 extracts road centerlines from RGB aerial imagery, including roads obscured by tree canopy. It uses a U-Net with a ResNet-18 encoder trained on 0.6 m NAIP imagery in the U.S. Mid-Atlantic.

The [MIT-licensed weights](https://huggingface.co/isaaccorley/chesapeakersc-v2) achieve 79.0% road pixel F1 and 53.5% canopy recall on 11,345 held-out county chips. On 6,385 Delaware chips, F1 is 82.6% and canopy recall is 66.1%. The [CC0 dataset](https://huggingface.co/datasets/isaaccorley/chesapeakersc-v2) provides the chip schema, manifests, and county splits.

## Inference

The pipeline resamples georeferenced uint8 RGB chips to approximately 0.6 m, divides pixel values by 255, and predicts road probabilities in 512 × 512 windows with 50% Hann overlap. It traces probability ridges with a 0.08 floor. Output is WGS84 GeoJSON LineStrings with class 1 and a confidence value equal to the mean road probability along the centerline before simplification.

Send north-up chips with a shared CRS in one request to connect roads across tile seams. Requests support up to 16 million pixels at 0.6 m. CPU inference uses ONNX Runtime.

## Fine-tuning

Fine-tuning starts from the RGB checkpoint and accepts same-zoom OAM-named chips with WGS84 road labels. Polygons rasterize directly. Centerlines use a configurable road width of 4 m by default. Labels should cover all roads in each chip because unlabeled pixels count as background.

Training and validation hold out whole spatial blocks and require at least two blocks. Validation reports road pixel F1 and IoU. The pipeline exports the fine-tuned model to ONNX. The supplied image uses CPU PyTorch. The shared model integration test uses synthetic road labels included with this model.

Follow the [fAIr contribution guide](https://hotosm.github.io/fAIr-models/contributing/model/) for installation, training, and serving commands.

## Limitations

Validate predictions for new regions and imagery sources. Canopy and junctions can produce missed roads or false connections. Mappers should review centerlines before adding them to OpenStreetMap or using them for routing.

## Licensing

Weights and model code use MIT. The dataset is CC0-1.0. The fAIr runtime uses AGPL-3.0-only.

Inference dependencies are pinned in requirements-inference.txt. ONNX Runtime and geomltoolkits use MIT. NumPy, SciPy, scikit-image, Rasterio, and Shapely use BSD-3-Clause. Training adds PyTorch and torchvision (BSD-3-Clause), segmentation-models-pytorch (MIT), and ONNX (Apache-2.0). fAIr also uses ZenML and MLflow (Apache-2.0).

## Citation

Caleb Robinson, Isaac Corley, Anthony Ortiz, Rahul Dodhia, Juan M. Lavista Ferres, and Peyman Najafirad. [Seeing the Roads Through the Trees](https://arxiv.org/abs/2401.06762), 2024.
