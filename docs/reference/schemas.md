# STAC Extension Schemas

fAIr defines three custom STAC extensions that layer fAIr-specific metadata on top of upstream extensions (MLM, Label, Version, Classification).

Each schema follows the [STAC Extension](https://stac-extensions.github.io/) pattern: a JSON Schema with a `$id` matching its published URL, a `stac_extensions` check, required properties, and field definitions.

## Base Model

Extends the [MLM Extension](https://stac-extensions.github.io/mlm/v1.5.1/schema.json) with training pipeline metadata: metrics spec, split spec, hyperparameter bounds, and runtime container references.

**Schema URL:** [`v1.0.0/base-model/schema.json`](../../v1.0.0/base-model/schema.json)

**Required properties:** `title`, `description`, `mlm:name`, `mlm:architecture`, `mlm:tasks`, `mlm:framework`, `mlm:framework_version`, `mlm:pretrained`, `mlm:input`, `mlm:output`, `mlm:hyperparameters`, `keywords`, `version`, `license`, `fair:metrics_spec`, `fair:split_spec`

**Required assets:** `model`, `source-code`, `mlm:training`, `mlm:inference`

## Dataset

Extends the [Label Extension](https://stac-extensions.github.io/label/v1.0.1/schema.json) with fAIr training data metadata: user attribution, chip counts, and download archives.

**Schema URL:** [`v1.0.0/dataset/schema.json`](../../v1.0.0/dataset/schema.json)

**Required properties:** `title`, `description`, `label:type`, `label:tasks`, `label:classes`, `keywords`, `fair:user_id`, `version`, `deprecated`

**Required assets:** `chips`, `labels`

## Local Model (Finetuned)

Extends the base model schema with training provenance: links to the base model and dataset, evaluation metrics, training duration, and ZenML artifact references.

**Schema URL:** [`v1.0.0/local-model/schema.json`](../../v1.0.0/local-model/schema.json)

**Required properties:** `title`, `description`, `mlm:name`, `mlm:architecture`, `mlm:tasks`, `mlm:framework`, `mlm:framework_version`, `mlm:pretrained`, `mlm:pretrained_source`, `mlm:input`, `mlm:output`, `mlm:hyperparameters`, `keywords`, `version`, `deprecated`, `fair:user_id`

**Required assets:** `model`, `source-code`

## Versioning and IDs

Each item type follows the [STAC Version Extension](https://github.com/stac-extensions/version) with a consistent archiving pattern.

### ID Strategy

| Type | ID | How determined |
|------|-----|----------------|
| Base model | Human-readable slug | `item_id` from the STAC JSON file (e.g. `yolo11n-detection`) |
| Dataset | Human-readable slug | `_slugify(title)` or `item_id` from the STAC JSON file |
| Local model | UUID | ZenML model version ID (unique per user/training run) |

Local models use UUIDs because the same base model + dataset pair can produce different finetuned models across users and training runs.

### Version Lifecycle

All three types use the `version` property (string, starting at `"1"`).

**Base models and datasets** follow the same pattern:

1. On first register: `version: "1"`, item ID is the slug (e.g. `yolo11n-detection`)
2. On re-register: previous item is archived as `{slug}-v{N}` with `deprecated: true`, new item keeps the original slug with `version: "{N+1}"`
3. Archived items link to their successor via `successor-version`; the active item links back via `predecessor-version`

Base models match on `mlm:name`, datasets match on `title` to find previous active versions.

**Local models** get their version from ZenML's model version number. No archiving is performed since each promotion creates a new item with a unique UUID.

### Version Links

All items use links from the [Version Extension](https://stac-extensions.github.io/version/v1.2.0/schema.json):

| Link relation | Direction | Present on |
|---------------|-----------|------------|
| `latest-version` | self-referencing | Active items only |
| `predecessor-version` | current -> previous | Items with version > 1 |
| `successor-version` | old -> new | Archived (deprecated) items |

### Example

After registering a dataset three times:

| Item ID | Version | Deprecated | Links |
|---------|---------|------------|-------|
| `buildings-banepa-segmentation-v1` | 1 | true | `successor-version` -> v2 |
| `buildings-banepa-segmentation-v2` | 2 | true | `successor-version` -> v3 |
| `buildings-banepa-segmentation` | 3 | false | `latest-version` -> self, `predecessor-version` -> v2 |

## Timestamps

Temporal tracking uses:

| Property | Set by | Purpose |
|----------|--------|---------|
| `created` | Builder (on first creation) | When the STAC item was first published |
| `updated` | Backend (on every publish) | When the item was last modified |

## Validation

Schemas are registered into PySTAC's `JsonSchemaSTACValidator.schema_cache` at runtime so `item.validate()` resolves them without network access:

```python
from fair.stac.validators import validate_item
import pystac

item = pystac.Item.from_file("models/yolo11n_detection/stac-item.json")
errors = validate_item(item)
```
