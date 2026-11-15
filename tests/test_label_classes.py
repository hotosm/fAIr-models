import pytest

from fair.stac.label_classes import (
    LabelClassError,
    ModelClass,
    adapt_features,
    dataset_classes,
    model_classes,
    resolve_classes,
)


def _model(classes: list[dict]) -> dict:
    return {"mlm:output": [{"name": "mask", "classification:classes": classes}]}


def _dataset(name: str, values: list[str]) -> dict:
    return {"label:properties": [name], "label:classes": [{"name": name, "classes": values}]}


_TREES = _model(
    [
        {"name": "background", "value": 0},
        {"name": "tree", "value": 1, "fair:osm_tags": {"natural": ["tree"]}},
    ]
)
_ROOFS = _model(
    [
        {"name": "metal_sheet", "value": 1},
        {"name": "thatch", "value": 2},
        {"name": "unknown", "value": 9, "nodata": True},
    ]
)


def test_model_classes_skip_nodata_and_keep_osm_tags() -> None:
    assert model_classes(_ROOFS) == [ModelClass("metal_sheet", 1), ModelClass("thatch", 2)]
    assert model_classes(_TREES)[1].osm_tags == {"natural": ["tree"]}


def test_model_without_classes_is_rejected() -> None:
    with pytest.raises(LabelClassError, match="no classification:classes"):
        model_classes({"mlm:output": [{"name": "mask"}]})


def test_dataset_classes_read_only_declared_label_properties() -> None:
    properties = {
        "label:properties": ["class"],
        "label:classes": [{"name": "class", "classes": ["thatch"]}, {"name": "other", "classes": ["x"]}],
    }
    assert dataset_classes(properties) == [("class", "thatch")]


def test_osm_tags_match_a_model_class() -> None:
    resolved = resolve_classes(_dataset("natural", ["tree"]), _TREES)
    assert resolved[("natural", "tree")].value == 1


def test_own_labels_match_by_class_name() -> None:
    resolved = resolve_classes(_dataset("class", ["thatch", "metal_sheet"]), _ROOFS)
    assert {key: cls.value for key, cls in resolved.items()} == {("class", "thatch"): 2, ("class", "metal_sheet"): 1}


def test_unmatched_dataset_classes_name_both_sides() -> None:
    with pytest.raises(LabelClassError, match=r"\['building=yes'\] match no model class \['background', 'tree'\]"):
        resolve_classes(_dataset("building", ["yes"]), _TREES)


def test_class_map_resolves_a_mismatch() -> None:
    resolved = resolve_classes(_dataset("roof", ["iron"]), _ROOFS, {"roof=iron": "metal_sheet"})
    assert resolved[("roof", "iron")].value == 1


@pytest.mark.parametrize(
    ("class_map", "message"),
    [
        ({"roof=tin": "metal_sheet"}, "'roof=tin' is not a dataset class"),
        ({"roof=iron": "steel"}, "'steel' is not a model class"),
    ],
)
def test_class_map_is_checked_against_both_items(class_map: dict, message: str) -> None:
    with pytest.raises(LabelClassError, match=message):
        resolve_classes(_dataset("roof", ["iron"]), _ROOFS, class_map)


def test_adapt_features_sets_the_model_value_and_keeps_properties() -> None:
    features = [
        {"type": "Feature", "properties": {"natural": "tree", "tags": {"diameter_crown": "4 m"}}, "geometry": None}
    ]
    adapted = adapt_features(features, _dataset("natural", ["tree"]), _TREES)
    assert adapted[0]["properties"] == {"natural": "tree", "tags": {"diameter_crown": "4 m"}, "label": 1}


def test_adapt_features_rejects_a_feature_without_a_dataset_class() -> None:
    features = [{"type": "Feature", "properties": {"class": "slate"}, "geometry": None}]
    with pytest.raises(LabelClassError, match="feature 0 carries none"):
        adapt_features(features, _dataset("class", ["thatch"]), _ROOFS)


def test_a_wildcard_beside_values_accepts_any_value() -> None:
    model = _model([{"name": "building", "value": 1, "fair:osm_tags": {"building": ["*", "yes"]}}])
    assert resolve_classes(_dataset("building", ["house"]), model)[("building", "house")].value == 1


def test_adapt_features_reads_the_declared_property_not_nested_tags() -> None:
    features = [{"type": "Feature", "properties": {"class": "thatch", "tags": {"x": "y"}}, "geometry": None}]
    assert adapt_features(features, _dataset("class", ["thatch"]), _ROOFS)[0]["properties"]["label"] == 2
