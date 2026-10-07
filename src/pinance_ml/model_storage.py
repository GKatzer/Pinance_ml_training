"""MinIO-backed model artifact storage -- the contract between
predictor-ml-training (this repo, writes) and predictor-ml-inference
(inference host, reads) for auto-retrain / shadow deployment (README "Ретрейн
регрессора").

Fixed per-symbol slots, not a version-numbered path per run:

    {bucket}/{symbol}/production/metadata.json
    {bucket}/{symbol}/production/h{1..12}.txt
    {bucket}/{symbol}/production/h{1..12}_q{quantile}.txt   (optional, see below)
    {bucket}/{symbol}/candidate/metadata.json
    {bucket}/{symbol}/candidate/h{1..12}.txt
    {bucket}/{symbol}/candidate/h{1..12}_q{quantile}.txt    (optional, see below)

the inference host's `/predict/{symbol}` and `/predict/{symbol}/shadow` always read the
same two keys, so it never needs to discover "what's the latest version"
via a bucket listing -- it just re-polls the same path and compares
metadata.json's model_version to whatever it already has loaded, hot-
reloading only on a change. The real version identity lives inside each
slot's metadata.json (export_models.py's model_version field), not the
object path -- promoting a candidate to production is a same-bucket
object copy (slot -> slot), not a new upload.

Confidence-corridor tails (README "доверительный коридор"): each slot can
additionally have `h{horizon}_q{quantile}.txt` for every (horizon,
quantile) pair in that slot's metadata.json `quantile_levels` list (e.g.
`quantile_levels: [0.1, 0.9]` -> h1_q0.1.txt, h1_q0.9.txt, ...
h12_q0.9.txt). The filename is fully determined by `quantile_horizons` x
`quantile_levels`, both already in metadata.json, so the inference
service never needs to list the bucket to discover what's there --
construct the expected paths and fetch them, same as it already does for
h{1..12}.txt. The median (alpha=0.5) is *not* a separate file:
`quantile_median_source: "point_model"` in metadata.json means
h{horizon}.txt (already loaded for the point forecast) doubles as the
corridor's center line -- objective="regression_l1"'s minimizer is the
median. `quantile_levels` absent or empty means this slot has no
corridor, same as the pre-corridor format.

Point models and the corridor are trained, gated, and pushed by two
entirely separate pipelines (scripts/auto_retrain.py and
scripts/auto_retrain_quantiles.py respectively, on independent
schedules/promotion criteria -- see either script's own docstring for
why) that can each update a slot at different times. A push from one
pipeline must never wipe the other's fields or files -- see
`merge_metadata` below, used by both before every write. The two
pipelines' metadata keys are disjoint by construction: unprefixed keys
(`schema_version`, `model_version`, `horizons`, ...) belong to the point
pipeline, `quantile_`-prefixed keys to the corridor pipeline.

Promotion (candidate -> production) mirrors that same split:
`promote_candidate_point` / `promote_candidate_quantiles` below each move
only their own pipeline's files (`h{h}.txt` vs `h{h}_q{q}.txt`) and
metadata keys, merging onto production's existing metadata.json instead
of overwriting it -- the same "don't wipe the other pipeline's stuff"
contract as a push into candidate, now also enforced on the way out.
`promote_candidate` (whole-slot copy, kept for callers that genuinely
want both at once) is what let a rejected point model ride into
production on an approved quantile push's coattails in a 2026-08-03
incident (see scripts/promote_if_better.py's module docstring) --
promote_if_better.py itself now always calls the two split functions
independently instead.

Rollback safety net: `previous` is a third fixed slot holding a byte-for-
byte snapshot of production as it was just BEFORE the most recent
promotion (`snapshot_production`, taken automatically by both
`promote_candidate_*` unless the caller already took one for a combined
promotion). `rollback_production` mirrors it back. One level deep on
purpose -- this is an undo button, not a version history; the real
version identity still lives in metadata.json. Note rolling back to a
model with a different feature schema is only useful together with
rolling back the feature code, otherwise predictor-ml-inference will
reject it for the same schema mismatch that motivated the promotion.

Consistency checks (`slot_consistency_problems`): a pure function over a
slot's metadata.json + object names -- every file the metadata promises
(h{h}.txt, h{h}_q{q}.txt) is present, schema_version really is the hash
of feature_columns, and the point/corridor halves don't disagree about
the schema. Run on the candidate before a promotion and on production
after it.
"""

import hashlib
import io
import json
import re
from functools import lru_cache
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from pinance_ml.config import MINIO_ACCESS_KEY, MINIO_ENDPOINT, MINIO_MODELS_BUCKET, MINIO_SECRET_KEY, MINIO_SECURE

PRODUCTION_SLOT = "production"
CANDIDATE_SLOT = "candidate"
PREVIOUS_SLOT = "previous"


@lru_cache(maxsize=1)
def _client() -> Minio:
    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE)
    if not client.bucket_exists(MINIO_MODELS_BUCKET):
        client.make_bucket(MINIO_MODELS_BUCKET)
    return client


def _prefix(symbol: str, slot: str) -> str:
    return f"{symbol}/{slot}/"


def upload_model_set(symbol: str, slot: str, source_dir: Path) -> None:
    """Upload every file in `source_dir` (export_models.py's per-symbol
    output: metadata.json + h{h}.txt, plus h{h}_q{q}.txt if
    --include-quantiles was used) to the given slot, overwriting
    whatever was there before -- slots are fixed-identity, not
    append-only/versioned storage."""
    client = _client()
    prefix = _prefix(symbol, slot)
    for file in sorted(source_dir.iterdir()):
        if file.is_file():
            client.fput_object(MINIO_MODELS_BUCKET, prefix + file.name, str(file))


def download_metadata(symbol: str, slot: str) -> dict | None:
    """Fetch just a slot's metadata.json, without downloading its model
    files -- used before a partial (point-only or quantile-only) push to
    see what's already there, so merge_metadata can update only that
    pipeline's own keys. None if the slot doesn't exist yet, same
    convention as download_model_set."""
    client = _client()
    prefix = _prefix(symbol, slot)
    try:
        response = client.get_object(MINIO_MODELS_BUCKET, prefix + "metadata.json")
    except S3Error:
        return None
    try:
        return json.loads(response.read())
    finally:
        response.close()
        response.release_conn()


def merge_metadata(existing: dict | None, updates: dict) -> dict:
    """Overlay `updates` onto `existing` (or an empty dict, if the slot
    has no metadata yet). The point and quantile pipelines' keys are
    disjoint by construction (see module docstring), so this is a plain
    shallow merge -- its only real job is being the one tested chokepoint
    for "a point-only push must never wipe an already-promoted corridor's
    quantile_* fields, or vice versa", rather than that behavior being
    reimplemented ad hoc in each script."""
    return {**(existing or {}), **updates}


def download_model_set(symbol: str, slot: str, dest_dir: Path) -> dict | None:
    """Download a slot's metadata.json + model files into `dest_dir`,
    returning the parsed metadata -- or None if that slot doesn't exist
    yet (auto_retrain.py's bootstrap case: no production pushed so far).
    """
    client = _client()
    prefix = _prefix(symbol, slot)

    try:
        objects = list(client.list_objects(MINIO_MODELS_BUCKET, prefix=prefix))
    except S3Error:
        return None
    if not objects:
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    for obj in objects:
        name = obj.object_name[len(prefix):]
        client.fget_object(MINIO_MODELS_BUCKET, obj.object_name, str(dest_dir / name))

    metadata_path = dest_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


def promote_candidate(symbol: str) -> None:
    """Copy the ENTIRE candidate slot onto the production slot -- point
    model and corridor together, whichever pipeline pushed each most
    recently, overwriting whatever was in production. Server-side copy
    (no download/re-upload round-trip).

    scripts/promote_if_better.py no longer calls this: it always goes
    through promote_candidate_point / promote_candidate_quantiles below
    instead, independently, so a promotion driven by one pipeline's gate
    can never also move the other pipeline's artifact as a side effect
    (see either split function's docstring for the 2026-08-03 incident
    that made this the default). Kept for callers that genuinely want a
    full-slot copy (e.g. manual/CLI bootstrap of a symbol from scratch)."""
    _copy_candidate_files(symbol, lambda name: True)


def _is_point_model_file(name: str) -> bool:
    """h{horizon}.txt -- the point/median model. Excludes both a
    quantile-corridor tail (h{horizon}_q{quantile}.txt, matched by
    _is_quantile_model_file instead) and metadata.json (handled
    separately, merged rather than copied wholesale)."""
    return re.fullmatch(r"h\d+\.txt", name) is not None


def _is_quantile_model_file(name: str) -> bool:
    """h{horizon}_q{quantile}.txt -- a confidence-corridor tail model."""
    return re.fullmatch(r"h\d+_q[\d.]+\.txt", name) is not None


def _split_metadata_by_pipeline(metadata: dict) -> tuple[dict, dict]:
    """Partition a slot's metadata.json into (point_keys, quantile_keys)
    -- same disjoint-keys contract merge_metadata already relies on (see
    module docstring): quantile_-prefixed keys belong to the corridor
    pipeline, everything else (including shared keys like `symbol`) to
    the point pipeline."""
    quantile_keys = {k: v for k, v in metadata.items() if k.startswith("quantile_")}
    point_keys = {k: v for k, v in metadata.items() if k not in quantile_keys}
    return point_keys, quantile_keys


def _copy_candidate_files(symbol: str, file_filter) -> None:
    client = _client()
    src_prefix = _prefix(symbol, CANDIDATE_SLOT)
    dst_prefix = _prefix(symbol, PRODUCTION_SLOT)

    from minio.commonconfig import CopySource

    objects = list(client.list_objects(MINIO_MODELS_BUCKET, prefix=src_prefix))
    if not objects:
        raise FileNotFoundError(f"{symbol}: no candidate model to promote")

    relevant = [obj for obj in objects if file_filter(obj.object_name[len(src_prefix):])]
    if not relevant:
        raise FileNotFoundError(f"{symbol}: candidate has no matching model files to promote")

    for obj in relevant:
        name = obj.object_name[len(src_prefix):]
        client.copy_object(MINIO_MODELS_BUCKET, dst_prefix + name, CopySource(MINIO_MODELS_BUCKET, obj.object_name))


def _put_metadata(symbol: str, slot: str, metadata: dict) -> None:
    payload = json.dumps(metadata, indent=2).encode()
    client = _client()
    client.put_object(
        MINIO_MODELS_BUCKET,
        _prefix(symbol, slot) + "metadata.json",
        io.BytesIO(payload),
        length=len(payload),
        content_type="application/json",
    )


def _expected_schema_version(feat_cols: list[str]) -> str:
    """Same hash as scripts/export_models.py's _schema_version (can't be
    imported from here -- scripts/ isn't a package; tests/test_model_
    storage.py pins the two together)."""
    return hashlib.sha256(",".join(feat_cols).encode()).hexdigest()[:12]


def slot_consistency_problems(
    metadata: dict | None, object_names: set[str], *, point: bool = True, quantiles: bool = True
) -> list[str]:
    """Everything wrong with a slot, as human-readable strings (empty list
    = consistent). `object_names` are bare file names in the slot (no
    prefix). `point` / `quantiles` select which pipeline's half to check
    -- promote_candidate_point only needs the point half of the candidate
    to be sound, and shouldn't be blocked by a half-written corridor.
    Mixed-schema (point vs corridor disagree) is checked only when both
    halves are checked and present."""
    if metadata is None:
        return ["metadata.json missing"]
    problems: list[str] = []

    if point:
        feat_cols = metadata.get("feature_columns")
        if not feat_cols:
            problems.append("point: feature_columns missing/empty")
        elif metadata.get("schema_version") != _expected_schema_version(feat_cols):
            problems.append(
                f"point: schema_version {metadata.get('schema_version')} != hash of feature_columns "
                f"{_expected_schema_version(feat_cols)}"
            )
        horizons = metadata.get("horizons") or []
        if not horizons:
            problems.append("point: horizons missing/empty")
        for h in horizons:
            if f"h{h}.txt" not in object_names:
                problems.append(f"point: h{h}.txt missing")
        if not metadata.get("model_version"):
            problems.append("point: model_version missing")

    levels = metadata.get("quantile_levels") or []
    if quantiles and levels:
        for h in metadata.get("quantile_horizons") or []:
            for q in levels:
                if f"h{h}_q{q}.txt" not in object_names:
                    problems.append(f"quantile: h{h}_q{q}.txt missing")
        if not metadata.get("quantile_model_version"):
            problems.append("quantile: quantile_model_version missing")
        if point and metadata.get("quantile_schema_version") != metadata.get("schema_version"):
            problems.append(
                f"mixed schema: point {metadata.get('schema_version')} vs corridor "
                f"{metadata.get('quantile_schema_version')}"
            )
    return problems


def check_slot_consistency(symbol: str, slot: str, *, point: bool = True, quantiles: bool = True) -> list[str]:
    """slot_consistency_problems against what's actually in MinIO."""
    client = _client()
    prefix = _prefix(symbol, slot)
    names = {obj.object_name[len(prefix):] for obj in client.list_objects(MINIO_MODELS_BUCKET, prefix=prefix)}
    return slot_consistency_problems(download_metadata(symbol, slot), names, point=point, quantiles=quantiles)


def _mirror_slot(symbol: str, src_slot: str, dst_slot: str) -> int:
    """Make `dst_slot` an exact copy of `src_slot` (server-side copies,
    plus deleting dst objects src doesn't have, so e.g. a corridor that
    exists only in dst doesn't survive a rollback). Returns the number of
    objects copied."""
    from minio.commonconfig import CopySource

    client = _client()
    src_prefix, dst_prefix = _prefix(symbol, src_slot), _prefix(symbol, dst_slot)
    src_objects = list(client.list_objects(MINIO_MODELS_BUCKET, prefix=src_prefix))
    if not src_objects:
        raise FileNotFoundError(f"{symbol}: slot '{src_slot}' is empty, nothing to mirror")
    src_names = set()
    for obj in src_objects:
        name = obj.object_name[len(src_prefix):]
        src_names.add(name)
        client.copy_object(MINIO_MODELS_BUCKET, dst_prefix + name, CopySource(MINIO_MODELS_BUCKET, obj.object_name))
    for obj in list(client.list_objects(MINIO_MODELS_BUCKET, prefix=dst_prefix)):
        if obj.object_name[len(dst_prefix):] not in src_names:
            client.remove_object(MINIO_MODELS_BUCKET, obj.object_name)
    return len(src_objects)


def snapshot_production(symbol: str) -> bool:
    """Save production as `previous` (overwriting the old snapshot).
    Returns False (nothing to save) if the symbol has no production yet."""
    if download_metadata(symbol, PRODUCTION_SLOT) is None:
        return False
    _mirror_slot(symbol, PRODUCTION_SLOT, PREVIOUS_SLOT)
    return True


def rollback_production(symbol: str) -> None:
    """Restore production from `previous`. Refuses if `previous` is
    inconsistent -- an undo that installs a broken model set is worse than
    no undo. Does NOT touch `previous` itself, so rolling back twice is a
    no-op rather than flipping between the two states."""
    problems = check_slot_consistency(symbol, PREVIOUS_SLOT)
    if problems:
        raise ValueError(f"{symbol}: refusing to roll back, 'previous' is inconsistent: {problems}")
    _mirror_slot(symbol, PREVIOUS_SLOT, PRODUCTION_SLOT)


def promote_candidate_point(symbol: str, snapshot: bool = True) -> None:
    """Promote only the candidate's point model (h{h}.txt files +
    unprefixed metadata keys) onto production -- production's existing
    corridor (files and quantile_-prefixed keys), if any, is left
    completely untouched, same as a quantile-only push into candidate
    already leaves the point side alone (see module docstring).

    Refuses (ValueError) if the candidate's point half is inconsistent,
    and snapshots production into `previous` first unless the caller
    already did (`snapshot=False`, for a combined point+corridor
    promotion that should leave ONE snapshot of the pre-promotion state)."""
    problems = check_slot_consistency(symbol, CANDIDATE_SLOT, point=True, quantiles=False)
    if problems:
        raise ValueError(f"{symbol}: candidate point model is inconsistent, not promoting: {problems}")
    if snapshot:
        snapshot_production(symbol)
    _copy_candidate_files(symbol, _is_point_model_file)
    candidate_metadata = download_metadata(symbol, CANDIDATE_SLOT)
    point_keys, _ = _split_metadata_by_pipeline(candidate_metadata)
    merged = merge_metadata(download_metadata(symbol, PRODUCTION_SLOT), point_keys)
    _put_metadata(symbol, PRODUCTION_SLOT, merged)


def promote_candidate_quantiles(symbol: str, snapshot: bool = True) -> None:
    """Promote only the candidate's confidence-corridor tail models
    (h{h}_q{q}.txt files + quantile_-prefixed metadata keys) onto
    production -- production's existing point model (model_version and
    everything else unprefixed) is left completely untouched.

    This is what closes the 2026-08-03 incident (BTCUSDT/BNBUSDT, see
    scripts/promote_if_better.py's module docstring) at the storage
    layer: a point model auto_retrain.py's own offline gate rejected had
    ridden into production bundled with an approved quantile push,
    because the only tool available at the time (promote_candidate,
    whole-slot) had no way to move one without the other.

    Same consistency refusal / `snapshot` contract as promote_candidate_point
    (corridor half of the candidate only)."""
    problems = check_slot_consistency(symbol, CANDIDATE_SLOT, point=False, quantiles=True)
    if problems:
        raise ValueError(f"{symbol}: candidate corridor is inconsistent, not promoting: {problems}")
    if snapshot:
        snapshot_production(symbol)
    _copy_candidate_files(symbol, _is_quantile_model_file)
    candidate_metadata = download_metadata(symbol, CANDIDATE_SLOT)
    _, quantile_keys = _split_metadata_by_pipeline(candidate_metadata)
    merged = merge_metadata(download_metadata(symbol, PRODUCTION_SLOT), quantile_keys)
    _put_metadata(symbol, PRODUCTION_SLOT, merged)
