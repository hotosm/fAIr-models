"""Flavor for Fair STAC Model Registry."""

from zenml.model_registries.base_model_registry import (
    BaseModelRegistryConfig,
    BaseModelRegistryFlavor,
)


class STACModelRegistryConfig(BaseModelRegistryConfig):
    """Config for STAC Model Registry."""

    catalog_path: str = "./stac_catalog"


class STACModelRegistryFlavor(BaseModelRegistryFlavor):
    """Flavor for STAC-based Model Registry."""

    @property
    def name(self) -> str:
        """Name of the flavor."""
        return "stac"

    @property
    def source(self) -> str:
        """Module source for the flavor."""
        return "fair_integrations.registry.stac:STACModelRegistry"

    @property
    def config_class(self):
        """Config class for the flavor."""
        return STACModelRegistryConfig
