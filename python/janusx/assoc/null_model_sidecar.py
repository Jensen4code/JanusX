"""Versioned GWAS null-model sidecar schema and matching helpers.

The sidecar is deliberately implemented with the Python standard library only.
It is embedded in the human-readable GWAS log between versioned sentinels so
future readers can ignore ordinary log text and older logs can remain valid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from typing import Any, Iterable, Mapping


SIDECAR_SCHEMA_V1 = "janusx.gwas-null-model/v1"
SIDECAR_BEGIN_V1 = "[JANUSX_GWAS_NULL_MODEL_V1_BEGIN]"
SIDECAR_END_V1 = "[JANUSX_GWAS_NULL_MODEL_V1_END]"

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_HASH_LENGTH = struct.Struct(">Q")
_HASH_CHUNK_SIZE = 1024 * 1024


class SidecarError(Exception):
    """Base class for sidecar discovery and format errors."""


class SidecarFormatError(SidecarError, ValueError):
    """Raised when a sidecar block is incomplete or violates its schema."""


class SidecarNotFound(SidecarError, LookupError):
    """Raised when no sidecar matches a result file."""


class SidecarAmbiguous(SidecarError, LookupError):
    """Raised when more than one sidecar matches a result file."""


@dataclass(frozen=True)
class FileFingerprintV1:
    canonical_path: str
    basename: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class GwasNullModelSidecarV1:
    schema: str
    created_at: str
    janusx_version: str
    result: FileFingerprintV1
    model: str
    trait: str
    z_columns: tuple[str, str]
    effective_snp_count: int
    genotype_prefix: str
    genotype_files: tuple[FileFingerprintV1, ...]
    phenotype_file: FileFingerprintV1
    phenotype_id_column: str
    phenotype_trait_column: str
    covariate_file: FileFingerprintV1 | None
    covariate_columns: tuple[str, ...]
    kinship_file: FileFingerprintV1
    kinship_id_file: FileFingerprintV1
    kinship_format: str
    kinship_shape: tuple[int, int]
    sample_count: int
    sample_order_sha256: str
    lambda_null: float
    sigma_g2: float | None
    sigma_e2: float | None
    pve: float
    grm_trace_mean: float
    fixed_effect_columns: tuple[str, ...]
    genotype_filters: dict[str, object]
    allele_coding: str


__all__ = [
    "FileFingerprintV1",
    "GwasNullModelSidecarV1",
    "SIDECAR_BEGIN_V1",
    "SIDECAR_END_V1",
    "SIDECAR_SCHEMA_V1",
    "SidecarAmbiguous",
    "SidecarError",
    "SidecarFormatError",
    "SidecarNotFound",
    "find_matching_sidecar",
    "fingerprint_file",
    "hash_ordered_sample_ids",
    "parse_sidecar_blocks",
    "serialize_sidecar_block",
]


def fingerprint_file(path: str | Path) -> FileFingerprintV1:
    """Return the canonical path, size, and streamed SHA-256 of *path*."""

    canonical = Path(path).expanduser().resolve(strict=True)
    stat_result = canonical.stat()
    digest = hashlib.sha256()
    with canonical.open("rb") as handle:
        while True:
            chunk = handle.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return FileFingerprintV1(
        canonical_path=str(canonical),
        basename=canonical.name,
        size_bytes=int(stat_result.st_size),
        sha256=digest.hexdigest(),
    )


def hash_ordered_sample_ids(sample_ids: Iterable[str]) -> str:
    """Hash sample IDs using an unambiguous 8-byte length-prefixed encoding."""

    digest = hashlib.sha256()
    for sample_id in sample_ids:
        if not isinstance(sample_id, str):
            raise TypeError("sample IDs must be strings")
        encoded = sample_id.encode("utf-8")
        digest.update(_HASH_LENGTH.pack(len(encoded)))
        digest.update(encoded)
    return digest.hexdigest()


def serialize_sidecar_block(record: GwasNullModelSidecarV1) -> str:
    """Serialize one v1 record as a deterministic sentinel-delimited block."""

    _validate_record(record)
    try:
        payload = asdict(record)
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SidecarFormatError(f"cannot serialize sidecar: {exc}") from exc
    return f"{SIDECAR_BEGIN_V1}\n{encoded}\n{SIDECAR_END_V1}\n"


def parse_sidecar_blocks(text: str) -> list[GwasNullModelSidecarV1]:
    """Parse all complete v1 blocks from ordinary log text."""

    if not isinstance(text, str):
        raise SidecarFormatError("sidecar log must be text")

    records: list[GwasNullModelSidecarV1] = []
    cursor = 0
    while True:
        begin = text.find(SIDECAR_BEGIN_V1, cursor)
        end_before_begin = text.find(SIDECAR_END_V1, cursor)
        if begin < 0:
            if end_before_begin >= 0:
                raise SidecarFormatError("sidecar end sentinel has no begin sentinel")
            break
        if end_before_begin >= 0 and end_before_begin < begin:
            raise SidecarFormatError("sidecar sentinels are out of order")

        content_start = begin + len(SIDECAR_BEGIN_V1)
        end = text.find(SIDECAR_END_V1, content_start)
        if end < 0:
            raise SidecarFormatError("sidecar begin sentinel has no end sentinel")
        raw_payload = text[content_start:end].strip()
        if not raw_payload:
            raise SidecarFormatError("sidecar block is empty")
        records.append(_parse_record(raw_payload))
        cursor = end + len(SIDECAR_END_V1)
    return records


def find_matching_sidecar(
    result_path: str | Path,
    candidates: Iterable[GwasNullModelSidecarV1],
) -> GwasNullModelSidecarV1:
    """Find one sidecar by canonical path, then moved-file fingerprint."""

    try:
        result_fingerprint = fingerprint_file(result_path)
    except (OSError, ValueError) as exc:
        raise SidecarNotFound(f"result file is unavailable: {result_path}") from exc

    records = list(candidates)
    for record in records:
        _validate_record(record)

    exact = [
        record
        for record in records
        if record.result.canonical_path == result_fingerprint.canonical_path
    ]
    if exact:
        return _one_match(exact, "canonical result path")

    moved = [
        record
        for record in records
        if (
            record.result.basename == result_fingerprint.basename
            and record.result.size_bytes == result_fingerprint.size_bytes
            and record.result.sha256.lower() == result_fingerprint.sha256.lower()
        )
    ]
    if moved:
        return _one_match(moved, "result basename, size, and SHA-256")
    raise SidecarNotFound(
        f"no sidecar matches result {result_fingerprint.canonical_path}"
    )


def _one_match(
    records: list[GwasNullModelSidecarV1], criterion: str
) -> GwasNullModelSidecarV1:
    if len(records) != 1:
        raise SidecarAmbiguous(
            f"{len(records)} sidecars match by {criterion}"
        )
    return records[0]


def _parse_record(raw_payload: str) -> GwasNullModelSidecarV1:
    try:
        payload = json.loads(
            raw_payload,
            object_pairs_hook=_object_pairs_without_duplicates,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except SidecarFormatError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SidecarFormatError(f"invalid sidecar JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SidecarFormatError("sidecar JSON must contain an object")

    try:
        record = GwasNullModelSidecarV1(
            schema=_string_field(payload, "schema"),
            created_at=_string_field(payload, "created_at"),
            janusx_version=_string_field(payload, "janusx_version"),
            result=_fingerprint_field(payload, "result"),
            model=_string_field(payload, "model"),
            trait=_string_field(payload, "trait"),
            z_columns=_string_tuple_field(payload, "z_columns", length=2),
            effective_snp_count=_positive_int_field(payload, "effective_snp_count"),
            genotype_prefix=_string_field(payload, "genotype_prefix"),
            genotype_files=_fingerprint_tuple_field(payload, "genotype_files"),
            phenotype_file=_fingerprint_field(payload, "phenotype_file"),
            phenotype_id_column=_string_field(payload, "phenotype_id_column"),
            phenotype_trait_column=_string_field(payload, "phenotype_trait_column"),
            covariate_file=_optional_fingerprint_field(payload, "covariate_file"),
            covariate_columns=_string_tuple_field(payload, "covariate_columns"),
            kinship_file=_fingerprint_field(payload, "kinship_file"),
            kinship_id_file=_fingerprint_field(payload, "kinship_id_file"),
            kinship_format=_string_field(payload, "kinship_format"),
            kinship_shape=_positive_shape_field(payload, "kinship_shape"),
            sample_count=_positive_int_field(payload, "sample_count"),
            sample_order_sha256=_string_field(payload, "sample_order_sha256"),
            lambda_null=_finite_number_field(payload, "lambda_null"),
            sigma_g2=_optional_finite_number_field(payload, "sigma_g2"),
            sigma_e2=_optional_finite_number_field(payload, "sigma_e2"),
            pve=_finite_number_field(payload, "pve"),
            grm_trace_mean=_finite_number_field(payload, "grm_trace_mean"),
            fixed_effect_columns=_string_tuple_field(payload, "fixed_effect_columns"),
            genotype_filters=_mapping_field(payload, "genotype_filters"),
            allele_coding=_string_field(payload, "allele_coding"),
        )
    except SidecarFormatError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise SidecarFormatError(f"invalid v1 sidecar fields: {exc}") from exc
    _validate_record(record)
    return record


def _object_pairs_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SidecarFormatError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise SidecarFormatError(f"non-finite JSON number: {value}")


def _require_field(payload: Mapping[str, Any], name: str) -> Any:
    if name not in payload:
        raise SidecarFormatError(f"missing sidecar field: {name}")
    return payload[name]


def _string_field(payload: Mapping[str, Any], name: str) -> str:
    value = _require_field(payload, name)
    if not isinstance(value, str):
        raise SidecarFormatError(f"sidecar field {name} must be a string")
    return value


def _string_tuple_field(
    payload: Mapping[str, Any], name: str, *, length: int | None = None
) -> tuple[str, ...]:
    value = _require_field(payload, name)
    if not isinstance(value, (list, tuple)):
        raise SidecarFormatError(f"sidecar field {name} must be an array")
    if length is not None and len(value) != length:
        raise SidecarFormatError(
            f"sidecar field {name} must contain exactly {length} values"
        )
    if any(not isinstance(item, str) for item in value):
        raise SidecarFormatError(f"sidecar field {name} must contain strings")
    return tuple(value)


def _fingerprint_field(
    payload: Mapping[str, Any], name: str
) -> FileFingerprintV1:
    value = _require_field(payload, name)
    if not isinstance(value, dict):
        raise SidecarFormatError(f"sidecar field {name} must be an object")
    try:
        fingerprint = FileFingerprintV1(
            canonical_path=_string_field(value, "canonical_path"),
            basename=_string_field(value, "basename"),
            size_bytes=_nonnegative_int_field(value, "size_bytes"),
            sha256=_string_field(value, "sha256"),
        )
    except SidecarFormatError:
        raise
    _validate_fingerprint(fingerprint, name)
    return fingerprint


def _optional_fingerprint_field(
    payload: Mapping[str, Any], name: str
) -> FileFingerprintV1 | None:
    value = _require_field(payload, name)
    if value is None:
        return None
    return _fingerprint_field(payload, name)


def _fingerprint_tuple_field(
    payload: Mapping[str, Any], name: str
) -> tuple[FileFingerprintV1, ...]:
    value = _require_field(payload, name)
    if not isinstance(value, (list, tuple)):
        raise SidecarFormatError(f"sidecar field {name} must be an array")
    if not value:
        raise SidecarFormatError(f"sidecar field {name} must not be empty")
    fingerprints: list[FileFingerprintV1] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise SidecarFormatError(
                f"sidecar field {name}[{index}] must be an object"
            )
        fingerprints.append(_fingerprint_field({name: item}, name))
    return tuple(fingerprints)


def _positive_int_field(payload: Mapping[str, Any], name: str) -> int:
    value = _nonnegative_int_field(payload, name)
    if value <= 0:
        raise SidecarFormatError(f"sidecar field {name} must be positive")
    return value


def _nonnegative_int_field(payload: Mapping[str, Any], name: str) -> int:
    value = _require_field(payload, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SidecarFormatError(f"sidecar field {name} must be an integer")
    if value < 0:
        raise SidecarFormatError(f"sidecar field {name} must not be negative")
    return value


def _positive_shape_field(
    payload: Mapping[str, Any], name: str
) -> tuple[int, int]:
    value = _require_field(payload, name)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SidecarFormatError(f"sidecar field {name} must have two dimensions")
    dimensions = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise SidecarFormatError(
                f"sidecar field {name} dimensions must be integers"
            )
        if dimension <= 0:
            raise SidecarFormatError(
                f"sidecar field {name} dimensions must be positive"
            )
        dimensions.append(dimension)
    return dimensions[0], dimensions[1]


def _finite_number_field(payload: Mapping[str, Any], name: str) -> float:
    value = _require_field(payload, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SidecarFormatError(f"sidecar field {name} must be numeric")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        finite = False
    if not finite:
        raise SidecarFormatError(f"sidecar field {name} must be finite")
    return value


def _optional_finite_number_field(
    payload: Mapping[str, Any], name: str
) -> float | None:
    value = _require_field(payload, name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SidecarFormatError(f"sidecar field {name} must be numeric or null")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        finite = False
    if not finite:
        raise SidecarFormatError(f"sidecar field {name} must be finite or null")
    return value


def _mapping_field(payload: Mapping[str, Any], name: str) -> dict[str, object]:
    value = _require_field(payload, name)
    if not isinstance(value, dict):
        raise SidecarFormatError(f"sidecar field {name} must be an object")
    return dict(value)


def _validate_fingerprint(
    fingerprint: FileFingerprintV1, field_name: str = "fingerprint"
) -> None:
    if not isinstance(fingerprint, FileFingerprintV1):
        raise SidecarFormatError(f"{field_name} must be a FileFingerprintV1")
    if not isinstance(fingerprint.canonical_path, str) or not fingerprint.canonical_path:
        raise SidecarFormatError(f"{field_name}.canonical_path must be a string")
    if not isinstance(fingerprint.basename, str) or not fingerprint.basename:
        raise SidecarFormatError(f"{field_name}.basename must be a string")
    if (
        isinstance(fingerprint.size_bytes, bool)
        or not isinstance(fingerprint.size_bytes, int)
        or fingerprint.size_bytes < 0
    ):
        raise SidecarFormatError(f"{field_name}.size_bytes must be non-negative")
    if not isinstance(fingerprint.sha256, str) or not _SHA256_RE.fullmatch(
        fingerprint.sha256
    ):
        raise SidecarFormatError(f"{field_name}.sha256 must be a 64-character hex hash")


def _validate_record(record: GwasNullModelSidecarV1) -> None:
    if not isinstance(record, GwasNullModelSidecarV1):
        raise SidecarFormatError("sidecar record must be GwasNullModelSidecarV1")
    if record.schema != SIDECAR_SCHEMA_V1:
        raise SidecarFormatError(f"unsupported sidecar schema: {record.schema!r}")
    for name in (
        "created_at",
        "janusx_version",
        "model",
        "trait",
        "genotype_prefix",
        "phenotype_id_column",
        "phenotype_trait_column",
        "kinship_format",
        "allele_coding",
    ):
        if not isinstance(getattr(record, name), str):
            raise SidecarFormatError(f"sidecar field {name} must be a string")

    _validate_fingerprint(record.result, "result")
    _validate_fingerprint(record.phenotype_file, "phenotype_file")
    _validate_fingerprint(record.kinship_file, "kinship_file")
    _validate_fingerprint(record.kinship_id_file, "kinship_id_file")
    if record.covariate_file is not None:
        _validate_fingerprint(record.covariate_file, "covariate_file")
    if not isinstance(record.genotype_files, (tuple, list)) or not record.genotype_files:
        raise SidecarFormatError("sidecar field genotype_files must not be empty")
    for index, fingerprint in enumerate(record.genotype_files):
        _validate_fingerprint(fingerprint, f"genotype_files[{index}]")

    _validate_string_sequence(record.z_columns, "z_columns", length=2)
    _validate_string_sequence(record.covariate_columns, "covariate_columns")
    _validate_string_sequence(record.fixed_effect_columns, "fixed_effect_columns")
    _validate_positive_int(record.effective_snp_count, "effective_snp_count")
    _validate_positive_int(record.sample_count, "sample_count")
    _validate_positive_shape(record.kinship_shape, "kinship_shape")
    _validate_hash(record.sample_order_sha256, "sample_order_sha256")
    for name in ("lambda_null", "pve", "grm_trace_mean"):
        _validate_finite_number(getattr(record, name), name)
    for name in ("sigma_g2", "sigma_e2"):
        value = getattr(record, name)
        if value is not None:
            _validate_finite_number(value, name)
    if not isinstance(record.genotype_filters, dict):
        raise SidecarFormatError("sidecar field genotype_filters must be an object")


def _validate_string_sequence(
    value: object, name: str, *, length: int | None = None
) -> None:
    if not isinstance(value, (tuple, list)):
        raise SidecarFormatError(f"sidecar field {name} must be an array")
    if length is not None and len(value) != length:
        raise SidecarFormatError(
            f"sidecar field {name} must contain exactly {length} values"
        )
    if any(not isinstance(item, str) for item in value):
        raise SidecarFormatError(f"sidecar field {name} must contain strings")


def _validate_positive_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SidecarFormatError(f"sidecar field {name} must be a positive integer")


def _validate_positive_shape(value: object, name: str) -> None:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 2
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in value
        )
    ):
        raise SidecarFormatError(f"sidecar field {name} must contain positive dimensions")


def _validate_hash(value: object, name: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise SidecarFormatError(f"sidecar field {name} must be a 64-character hex hash")


def _validate_finite_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SidecarFormatError(f"sidecar field {name} must be numeric")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        finite = False
    if not finite:
        raise SidecarFormatError(f"sidecar field {name} must be finite")
