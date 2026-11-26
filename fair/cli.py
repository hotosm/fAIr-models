"""`fair` command-line interface: a thin operator wrapper over `FairClient`.

Replicates the backend flow: build a draft dataset STAC item from a TMS URL + AOI,
register it, finetune a base model, promote the result, and patch STAC items in
place. Config comes from FAIR_* env vars (see `_client`); training and promotion
run through ZenML on its own stack, not this CLI.
"""

import json
import os
from pathlib import Path
from typing import Annotated, Any

import pystac
import typer

from fair.client import FairClient
from fair.datasets import DEFAULT_CLASS_PROPERTY, PRE_INPUT, materialize_dataset, preview_sample
from fair.infra import knative, registry, stack
from fair.stac.builders import build_dataset_item
from fair.stac.constants import (
    BASE_MODELS_COLLECTION,
    DATASETS_COLLECTION,
    LOCAL_MODELS_COLLECTION,
)
from fair.stac.inputs import IMAGE_INPUT

app = typer.Typer(no_args_is_help=True, add_completion=False, help="fAIr model-ops CLI.")
dataset_app = typer.Typer(no_args_is_help=True, help="Build and register training datasets.")
model_app = typer.Typer(no_args_is_help=True, help="Finetune and promote local models.")
basemodel_app = typer.Typer(no_args_is_help=True, help="Register base models.")
item_app = typer.Typer(no_args_is_help=True, help="Read and update STAC items in any collection.")
stack_app = typer.Typer(no_args_is_help=True, help="Bring the local Compose stack up and down.")
knative_app = typer.Typer(no_args_is_help=True, help="Manage the KNative service that serves a model.")
app.add_typer(dataset_app, name="dataset")
app.add_typer(model_app, name="model")
app.add_typer(basemodel_app, name="basemodel")
app.add_typer(item_app, name="item")
app.add_typer(stack_app, name="stack")
app.add_typer(knative_app, name="knative")

_NAMESPACE_HELP = "KNative namespace (default: $FAIR_KNATIVE_NAMESPACE, else 'predict')"

_COLLECTIONS = {BASE_MODELS_COLLECTION, DATASETS_COLLECTION, LOCAL_MODELS_COLLECTION}


def _client() -> FairClient:
    return FairClient(
        stac_api_url=os.environ.get("FAIR_STAC_API_URL"),
        stac_api_key=os.environ.get("FAIR_STAC_API_KEY"),
        dsn=os.environ.get("FAIR_DSN"),
        user_id=os.environ.get("FAIR_USER_ID", "anonymous"),
        upload_artifacts=os.environ.get("FAIR_UPLOAD_ARTIFACTS", "").lower() == "true",
    )


def _resolve_collection(name: str) -> str:
    if name not in _COLLECTIONS:
        raise typer.BadParameter(f"unknown collection {name!r}; expected one of {sorted(_COLLECTIONS)}")
    return name


def _parse_value(raw: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _dump_item(item: pystac.Item) -> str:
    return json.dumps(item.to_dict(transform_hrefs=False), indent=2)


RAW_DATA_API_URL = "https://api-prod.raw-data.hotosm.org/v1"


def _parse_osm_tags(osm_tags: str) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for entry in (part.strip() for part in osm_tags.split(",")):
        if entry:
            key, _, value = entry.partition("=")
            values = parsed.setdefault(key, [])
            if "*" not in values:
                values[:] = ["*"] if not value else [*values, value]
    return parsed


def _parse_class_map(raw: str) -> dict[str, str]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"--class-map is not JSON: {exc}") from exc
    if not isinstance(parsed, dict) or not all(isinstance(v, str) for v in parsed.values()):
        raise typer.BadParameter('--class-map must map "property=value" to a model class name')
    return parsed


@dataset_app.command("build")
def dataset_build(
    tms: Annotated[str | None, typer.Option(help="TMS/XYZ tile URL template")] = None,
    aoi: Annotated[Path | None, typer.Option(exists=True, dir_okay=False, help="AOI GeoJSON file")] = None,
    zoom: Annotated[int | None, typer.Option(help="Tile zoom level")] = None,
    from_preview: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="Model stac-item.json; imagery, zoom and area come from fair:preview"
        ),
    ] = None,
    tiles: Annotated[int, typer.Option(min=1, help="With --from-preview: tile square side")] = 8,
    out: Annotated[Path, typer.Option(help="Directory for downloaded assets + draft")] = Path("drafts"),
    title: Annotated[str | None, typer.Option(help="Dataset title (default: derived from AOI or model)")] = None,
    osm_tags: Annotated[
        str | None,
        typer.Option(help="OSM tag keys or key=value pairs, comma-separated (default: the model's fair:osm_tags)"),
    ] = None,
    geometry_type: Annotated[str | None, typer.Option(help="point | line | polygon")] = None,
    tasks: Annotated[str | None, typer.Option(help="Comma-separated label tasks")] = None,
    keywords: Annotated[str | None, typer.Option(help="Comma-separated keywords")] = None,
    osm_url: Annotated[
        str | None, typer.Option(help="raw-data API URL (default: $FAIR_RAW_DATA_API_URL, else HOT's public API)")
    ] = None,
    labels: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False, help="Own labels GeoJSON with a class per feature"),
    ] = None,
    class_property: Annotated[str, typer.Option(help="Property of --labels that holds the class")] = (
        DEFAULT_CLASS_PROPERTY
    ),
    labels_license: Annotated[str | None, typer.Option(help="SPDX license of --labels")] = None,
) -> None:
    """Download chips + labels (OSM or --labels) for an area and write an editable draft STAC item."""
    if from_preview is not None:
        model_item = json.loads(from_preview.read_text())
        sample = preview_sample(model_item, tiles)
        tms_urls = sample.tms_urls
        if tms is not None:
            if len(tms_urls) != 1:
                raise typer.BadParameter(f"--tms replaces one input's imagery; the model has {sorted(tms_urls)}")
            tms_urls = {name: tms for name in tms_urls}
        resolved_zoom = zoom or sample.zoom
        aoi_geojson = json.loads(aoi.read_text()) if aoi else sample.aoi
        if geometry_type is None and len(sample.geometry_types) > 1:
            raise typer.BadParameter(f"the model accepts {sample.geometry_types}; pick one with --geometry-type")
        if geometry_type is not None and geometry_type not in sample.geometry_types:
            raise typer.BadParameter(f"the model accepts {sample.geometry_types} labels, not {geometry_type}")
        default_tasks, default_geometry = sample.tasks, sample.geometry_types[0]
        default_osm_tags, default_keywords = sample.osm_tags, sample.keywords
        resolved_title = title or f"{model_item['id']}-sample"
    else:
        if tms is None or aoi is None or zoom is None:
            raise typer.BadParameter("--tms, --aoi and --zoom are required without --from-preview")
        tms_urls = {IMAGE_INPUT: tms}
        resolved_zoom = zoom
        aoi_geojson = json.loads(aoi.read_text())
        default_tasks, default_geometry = ["semantic-segmentation"], "polygon"
        default_osm_tags, default_keywords = {"building": ["*"]}, ["building"]
        resolved_title = title or f"{aoi.stem}-{geometry_type or default_geometry}"
    resolved_geometry = geometry_type or default_geometry
    label_tasks = [t.strip() for t in tasks.split(",") if t.strip()] if tasks else default_tasks

    if labels is not None and osm_tags is not None:
        raise typer.BadParameter("--osm-tags selects OSM labels; drop it when passing --labels")
    if labels is None and class_property != DEFAULT_CLASS_PROPERTY:
        raise typer.BadParameter("--class-property names a property of --labels")
    if labels is not None:
        if labels_license is None:
            raise typer.BadParameter("--labels needs --labels-license (an SPDX identifier)")
        own_labels: dict[str, Any] | None = json.loads(labels.read_text())
        osm_api, tag_filter = None, None
        license_id = labels_license
    else:
        tag_filter = _parse_osm_tags(osm_tags) if osm_tags else default_osm_tags
        if not tag_filter:
            raise typer.BadParameter("the model declares no fair:osm_tags; pass --osm-tags or --labels")
        osm_api = osm_url or os.environ.get("FAIR_RAW_DATA_API_URL", RAW_DATA_API_URL)
        own_labels = None
        license_id = "ODbL-1.0"
    keyword_list: list[str] = (
        [k.strip() for k in keywords.split(",") if k.strip()]
        if keywords
        else [*dict.fromkeys([*default_keywords, *label_tasks, resolved_geometry])]
    )
    user_id = os.environ.get("FAIR_USER_ID", "anonymous")
    data_dir = out if from_preview is not None else out / resolved_title

    built = materialize_dataset(
        tms_urls=tms_urls,
        aoi=aoi_geojson,
        zoom=resolved_zoom,
        out_dir=str(data_dir),
        geometry_type=resolved_geometry,
        osm_tags=tag_filter,
        osm_api_url=osm_api,
        labels=own_labels,
        class_property=class_property,
    )

    item = build_dataset_item(
        label_type="vector",
        label_tasks=label_tasks,
        label_classes=built.label_classes,
        label_properties=built.label_properties,
        keywords=keyword_list,
        chips_hrefs={name: os.path.relpath(path, out) for name, path in built.chips_dirs.items()},
        labels_href=os.path.relpath(built.labels_file, out),
        title=resolved_title,
        description="",
        user_id=user_id,
        providers=[{"name": user_id, "roles": ["producer"]}],
        geometry=built.geometry,
        bbox=built.bbox,
        source_imagery=next(url for name, url in tms_urls.items() if name != PRE_INPUT),
        imagery_hrefs=tms_urls,
        geometry_type=resolved_geometry,  # type: ignore[arg-type]
        license_id=license_id,
    )

    out.mkdir(parents=True, exist_ok=True)
    draft = out / ("stac-item.json" if from_preview is not None else f"dataset-{item.id}.json")
    draft.write_text(_dump_item(item))
    typer.echo(f"wrote draft {draft}")
    typer.echo(f"edit title/description/keywords, then: fair dataset register {draft}")


@dataset_app.command("register")
def dataset_register(
    draft: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Draft dataset STAC item JSON")],
) -> None:
    """Register an edited draft dataset STAC item."""
    typer.echo(_client().register_dataset(str(draft)))


@model_app.command("train")
def model_train(
    base: Annotated[str, typer.Option(help="Base model STAC id")],
    dataset: Annotated[str, typer.Option(help="Dataset STAC id")],
    name: Annotated[str, typer.Option(help="Model name")],
    epochs: Annotated[int | None, typer.Option()] = None,
    batch_size: Annotated[int | None, typer.Option()] = None,
    learning_rate: Annotated[float | None, typer.Option()] = None,
    samples_per_epoch: Annotated[int | None, typer.Option()] = None,
    chip_size: Annotated[int | None, typer.Option()] = None,
    class_map: Annotated[
        str | None,
        typer.Option(help='JSON mapping dataset classes to model classes, e.g. \'{"natural=palm": "tree"}\''),
    ] = None,
) -> None:
    """Finetune a base model on a dataset (blocks until the training run finishes)."""
    overrides: dict[str, object] = {}
    for key, value in (
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("learning_rate", learning_rate),
        ("samples_per_epoch", samples_per_epoch),
        ("chip_size", chip_size),
    ):
        if value is not None:
            overrides[key] = value

    model_name = _client().finetune(
        base_model_id=base,
        dataset_id=dataset,
        model_name=name,
        overrides=overrides or None,
        class_map=_parse_class_map(class_map) if class_map else None,
    )
    typer.echo(model_name)


@model_app.command("promote")
def model_promote(
    name: Annotated[str, typer.Argument(help="Trained model name (from `fair model train`)")],
    title: Annotated[str | None, typer.Option(help="Local model title")] = None,
    description: Annotated[str, typer.Option(help="Local model description")] = "",
    keywords: Annotated[str | None, typer.Option(help="Comma-separated keywords")] = None,
    base: Annotated[str | None, typer.Option(help="Base model STAC id (resolved from run if omitted)")] = None,
    dataset: Annotated[str | None, typer.Option(help="Dataset STAC id (resolved from run if omitted)")] = None,
    out: Annotated[Path | None, typer.Option(help="Write the published item here for further editing")] = None,
) -> None:
    """Publish a trained model to the local-models collection and make it available."""
    client = _client()
    keyword_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else None
    item_id = client.promote(
        name,
        base_model_id=base,
        dataset_id=dataset,
        title=title,
        description=description,
        keywords=keyword_list,
    )
    typer.echo(item_id)
    if out is not None:
        item = client._get_backend().get_item(LOCAL_MODELS_COLLECTION, item_id)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"model-{item_id}.json"
        path.write_text(_dump_item(item))
        typer.echo(f"wrote {path}")


@basemodel_app.command("register")
def basemodel_register(
    item: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Base model STAC item JSON")],
) -> None:
    """Register a base model STAC item."""
    typer.echo(_client().register_base_model(str(item)))


@basemodel_app.command("pin")
def basemodel_pin(
    item: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Base model STAC item JSON")],
) -> None:
    """Pin runtime image tags to digests in place."""
    data = json.loads(item.read_text())
    for key in registry.IMAGE_ASSET_KEYS:
        asset = data.get("assets", {}).get(key)
        if asset and "://" not in asset["href"]:
            asset["href"] = registry.resolve_digest(asset["href"])
            typer.echo(f"{key}: {asset['href']}")
    item.write_text(json.dumps(data, indent=2) + "\n")


@knative_app.command("register")
def knative_register(
    item: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Model STAC item JSON")],
    namespace: Annotated[str | None, typer.Option(help=_NAMESPACE_HELP)] = None,
) -> None:
    """Create or update a model's KNative service."""
    stac_item = pystac.Item.from_file(str(item))
    knative.ensure_knative_service(stac_item, namespace=namespace)
    typer.echo(knative.knative_service_name(stac_item.properties.get("mlm:name") or stac_item.id))


@knative_app.command("status")
def knative_status(
    name: Annotated[str, typer.Argument(help="Model name (mlm:name) or STAC item id")],
    namespace: Annotated[str | None, typer.Option(help=_NAMESPACE_HELP)] = None,
) -> None:
    """Show a model service's readiness and URL."""
    ready, url = knative.knative_service_status(name, namespace=namespace)
    typer.echo(f"ready={ready}\turl={url}")


@knative_app.command("reconcile")
def knative_reconcile(
    prune: Annotated[bool, typer.Option(help="Release services for models no longer in STAC")] = False,
    namespace: Annotated[str | None, typer.Option(help=_NAMESPACE_HELP)] = None,
) -> None:
    """Match KNative services to active STAC base models."""
    result = _client().reconcile_knative(knative_namespace=namespace, prune=prune)
    typer.echo(json.dumps(result))
    if result["failed"]:
        raise typer.Exit(1)


@knative_app.command("delete")
def knative_delete(
    name: Annotated[str, typer.Argument(help="Model name (mlm:name) or STAC item id")],
    namespace: Annotated[str | None, typer.Option(help=_NAMESPACE_HELP)] = None,
) -> None:
    """Delete a model's KNative service."""
    knative.delete_knative_service(name, namespace=namespace)
    typer.echo(f"deleted {knative.knative_service_name(name)}")


@item_app.command("get")
def item_get(
    collection: Annotated[str, typer.Argument(help="base-models | datasets | local-models")],
    item_id: Annotated[str, typer.Argument()],
    out: Annotated[Path | None, typer.Option(help="Write JSON here instead of stdout")] = None,
) -> None:
    """Fetch a STAC item and print it (or write it to a file)."""
    item = _client()._get_backend().get_item(_resolve_collection(collection), item_id)
    data = _dump_item(item)
    if out is not None:
        out.write_text(data)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(data)


@item_app.command("list")
def item_list(
    collection: Annotated[str, typer.Argument(help="base-models | datasets | local-models")],
    limit: Annotated[int | None, typer.Option(help="Max items to return")] = None,
) -> None:
    """List items in a collection (id, version, title)."""
    items = _client()._get_backend().list_items(_resolve_collection(collection), limit=limit)
    for it in items:
        flag = " [deprecated]" if it.properties.get("deprecated") else ""
        typer.echo(f"{it.id}\tv{it.properties.get('version', '?')}\t{it.properties.get('title', '')}{flag}")


@item_app.command("patch")
def item_patch(
    collection: Annotated[str, typer.Argument(help="base-models | datasets | local-models")],
    item_id: Annotated[str, typer.Argument()],
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", help="property as key=value (value parsed as JSON, else string); repeatable"),
    ] = None,
    file: Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False, help="merge-patch JSON body (full control)")
    ] = None,
) -> None:
    """Update a STAC item in place via an RFC 7396 merge-patch."""
    if file is not None and set_:
        raise typer.BadParameter("use either --set or --file, not both")
    if file is not None:
        patch = json.loads(file.read_text())
    elif set_:
        properties: dict[str, object] = {}
        for pair in set_:
            key, sep, raw = pair.partition("=")
            if not sep:
                raise typer.BadParameter(f"--set expects key=value, got {pair!r}")
            properties[key] = _parse_value(raw)
        patch = {"properties": properties}
    else:
        raise typer.BadParameter("provide --set or --file")

    item = _client()._get_backend().patch_item(_resolve_collection(collection), item_id, patch)
    typer.echo(f"patched {collection}/{item.id}")


@item_app.command("deprecate")
def item_deprecate(
    collection: Annotated[str, typer.Argument(help="base-models | datasets | local-models")],
    item_id: Annotated[str, typer.Argument()],
) -> None:
    """Mark a STAC item deprecated."""
    item = _client()._get_backend().deprecate_item(_resolve_collection(collection), item_id)
    typer.echo(f"deprecated {collection}/{item.id}")


@stack_app.command("up")
def stack_up() -> None:
    """Start Postgres + MinIO + STAC + MLflow + ZenML and register the ZenML stack."""
    stack.up()
    typer.echo("stack up. ZenML :8080  MLflow :5000  STAC :8082  MinIO :9001")


@stack_app.command("down")
def stack_down(
    volumes: Annotated[bool, typer.Option("--volumes/--keep", help="Also remove volumes and state")] = False,
) -> None:
    """Stop the stack (containers only, or --volumes to wipe state)."""
    stack.down(volumes=volumes)
    typer.echo("stack down")


@stack_app.command("status")
def stack_status() -> None:
    """Show the status of the stack containers."""
    stack.status()


if __name__ == "__main__":
    app()
