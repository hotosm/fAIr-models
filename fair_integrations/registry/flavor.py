"""Flavor for Fair STAC Model Registry."""

from typing import Type

from zenml.model_registries.base_model_registry import (
    BaseModelRegistryConfig,
    BaseModelRegistryFlavor,
)
from zenml.stack import StackComponent


class STACModelRegistryConfig(BaseModelRegistryConfig):
    """Config for STAC Model Registry."""

    catalog_path: str = "./stac_catalog"


class STACModelRegistryFlavor(BaseModelRegistryFlavor):
    """Flavor for STAC-based Model Registry."""

    @property
    def name(self) -> str:
        return "stac"

    @property
    def implementation_class(self) -> Type[StackComponent]:
        from fair_integrations.registry.stac import STACModelRegistry

        return STACModelRegistry

    @property
    def config_class(self) -> Type[STACModelRegistryConfig]:
        return STACModelRegistryConfig
