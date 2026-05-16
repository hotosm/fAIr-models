"""Lower the verbosity of known-noisy upstream loggers.

Call `quiet_third_party_loggers()` from app entrypoints, not library import:
log levels are global.
"""

import logging

_NOISY: dict[str, int] = {
    # boto3 logs "Found credentials in environment variables" per session init.
    "botocore.credentials": logging.WARNING,
    # rasterio logs "Skipping source" per non-intersecting tile during sampling.
    "rasterio.merge": logging.WARNING,
    "rasterio.stack": logging.WARNING,
    # httpx/httpcore log every HTTP request at INFO.
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    # MLflow auto-instruments Bedrock at startup; unused here.
    "mlflow.bedrock": logging.WARNING,
}


def quiet_third_party_loggers() -> None:
    for name, level in _NOISY.items():
        logging.getLogger(name).setLevel(level)
