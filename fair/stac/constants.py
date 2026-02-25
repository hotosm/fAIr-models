OCI_IMAGE_INDEX_TYPE = "application/vnd.oci.image.index.v1+json"

MLM_SCHEMA = "https://stac-extensions.github.io/mlm/v1.5.1/schema.json"
VERSION_SCHEMA = "https://stac-extensions.github.io/version/v1.2.0/schema.json"
CLASSIFICATION_SCHEMA = "https://stac-extensions.github.io/classification/v2.0.0/schema.json"
FILE_SCHEMA = "https://stac-extensions.github.io/file/v2.1.0/schema.json"
LABEL_SCHEMA = "https://stac-extensions.github.io/label/v1.0.1/schema.json"
RASTER_SCHEMA = "https://stac-extensions.github.io/raster/v1.1.0/schema.json"

MODEL_EXTENSIONS = [MLM_SCHEMA, VERSION_SCHEMA, CLASSIFICATION_SCHEMA, FILE_SCHEMA, RASTER_SCHEMA]
DATASET_EXTENSIONS = [LABEL_SCHEMA, FILE_SCHEMA]

BASE_MODELS_COLLECTION = "base-models"
LOCAL_MODELS_COLLECTION = "local-models"
DATASETS_COLLECTION = "datasets"
