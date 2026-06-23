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
from fair.datasets import materialize_dataset
from fair.stac.builders import build_dataset_item
from fair.stac.constants import (
    BASE_MODELS_COLLECTION,
    DATASETS_COLLECTION,
    LOCAL_MODELS_COLLECTION,
)

app = typer.Typer(no_args_is_help=True, add_completion=False, help="fAIr model-ops CLI.")
dataset_app = typer.Typer(no_args_is_help=True, help="Build and register training datasets.")
model_app = typer.Typer(no_args_is_help=True, help="Finetune and promote local models.")
basemodel_app = typer.Typer(no_args_is_help=True, help="Register base models.")
item_app = typer.Typer(no_args_is_help=True, help="Read and update STAC items in any collection.")
app.add_typer(dataset_app, name="dataset")
app.add_typer(model_app, name="model")
app.add_typer(basemodel_app, name="basemodel")
app.add_typer(item_app, name="item")

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


@dataset_app.command("build")
def dataset_build(
    tms: Annotated[str, typer.Option(help="TMS/XYZ tile URL template")],
    aoi: Annotated[Path, typer.Option(exists=True, dir_okay=False, help="AOI GeoJSON file")],
    zoom: Annotated[int, typer.Option(help="Tile zoom level")],
    out: Annotated[Path, typer.Option(help="Directory for downloaded assets + draft")] = Path("drafts"),
    title: Annotated[str | None, typer.Option(help="Dataset title (default: derived from AOI filename)")] = None,
    classes: Annotated[str, typer.Option(help="Comma-separated OSM tag keys to label")] = "building",
    geometry_type: Annotated[str, typer.Option(help="point | line | polygon")] = "polygon",
    tasks: Annotated[str, typer.Option(help="Comma-separated label tasks")] = "semantic-segmentation",
    keywords: Annotated[str | None, typer.Option(help="Comma-separated keywords")] = None,
    osm_url: Annotated[str | None, typer.Option(help="raw-data API URL (default: $FAIR_RAW_DATA_API_URL)")] = None,
) -> None:
    """Download chips + OSM labels for an AOI and write an editable draft STAC item."""
    osm_api = osm_url or os.environ.get("FAIR_RAW_DATA_API_URL")
    if not osm_api:
        raise typer.BadParameter("raw-data API URL required via --osm-url or $FAIR_RAW_DATA_API_URL")

    label_classes: list[dict[str, Any]] = [
        {"name": c.strip(), "classes": ["*"]} for c in classes.split(",") if c.strip()
    ]
    label_tasks = [t.strip() for t in tasks.split(",") if t.strip()]
    keyword_list: list[str] = (
        [k.strip() for k in keywords.split(",") if k.strip()]
        if keywords
        else [str(cls["name"]) for cls in label_classes]
    )
    resolved_title = title or f"{aoi.stem}-{geometry_type}"
    user_id = os.environ.get("FAIR_USER_ID", "anonymous")

    chips_href, labels_href, geometry, bbox = materialize_dataset(
        tms_url=tms,
        aoi=json.loads(aoi.read_text()),
        zoom=zoom,
        out_dir=str(out / resolved_title),
        label_classes=label_classes,
        geometry_type=geometry_type,
        osm_api_url=osm_api,
    )

    item = build_dataset_item(
        label_type="vector",
        label_tasks=label_tasks,
        label_classes=label_classes,
        keywords=keyword_list,
        chips_href=chips_href,
        labels_href=labels_href,
        title=resolved_title,
        description="",
        user_id=user_id,
        providers=[{"name": user_id, "roles": ["producer"]}],
        geometry=geometry,
        bbox=bbox,
        source_imagery=tms,
        source_imagery_href=tms,
        geometry_type=geometry_type,  # type: ignore[arg-type]
    )

    out.mkdir(parents=True, exist_ok=True)
    draft = out / f"dataset-{item.id}.json"
    draft.write_text(_dump_item(item))
    typer.echo(f"wrote draft {draft}")
    typer.echo(f"edit title/description/keywords/license, then: fair dataset register {draft}")


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


if __name__ == "__main__":
    app()
