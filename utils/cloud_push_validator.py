"""
Aurora Cloud Push Validator — utils/cloud_push_validator.py
============================================================
Provider-aware credential pre-validation and friendly error mapping
for AWS S3, Google Cloud Storage, and Microsoft Azure Blob Storage.

NON-INTERFERENCE RULE:
  This module does NOT modify cloud connectors, upload logic,
  or storage internals. It only performs pre-validation and error mapping.
"""

import datetime
import re
from typing import Dict, List, Tuple, Any


# ── Required Credential Fields (per provider) ────────────────────────────────

_REQUIRED_FIELDS = {
    "AWS S3": ["access_key", "secret_key", "region"],
    "GCS": ["service_account_json"],
    "Azure Blob": ["connection_string"],
}

_REQUIRED_PUSH_FIELDS = {
    "AWS S3": ["access_key", "secret_key", "region"],
    "GCS": ["service_account_json"],
    "Azure Blob": ["connection_string"],
}


# ── Friendly Error Mapping (per provider) ─────────────────────────────────────

_ERROR_MAP = {
    "AWS S3": [
        ("InvalidAccessKeyId",        "Invalid Access Key. Please verify your AWS access key."),
        ("SignatureDoesNotMatch",      "Invalid Secret Key. Please verify your AWS secret key."),
        ("NoSuchBucket",              "Bucket not found. Please verify the bucket name."),
        ("AccessDenied",              "Access denied. Check your IAM permissions."),
        ("NetworkingError",           "Region or network issue. Check your internet and AWS region."),
        ("ExpiredToken",              "Session token expired. Please refresh credentials."),
        ("RequestTimeTooSkewed",      "System clock is out of sync. Check your system time."),
    ],
    "GCS": [
        ("PERMISSION_DENIED",         "Service account lacks permission. Check IAM roles in GCP."),
        ("NOT_FOUND",                 "Bucket not found. Verify bucket name and project."),
        ("NotFound",                  "Bucket not found. Verify bucket name and project."),
        ("InvalidArgument",           "Invalid project or bucket configuration."),
        ("UNAUTHENTICATED",           "Service account credentials are invalid."),
        ("Forbidden",                 "Access forbidden. Check bucket permissions."),
    ],
    "Azure Blob": [
        ("AuthenticationFailed",      "Invalid Account Key. Please verify your Azure storage key."),
        ("ContainerNotFound",         "Container not found. Verify container name."),
        ("AuthorizationPermissionMismatch", "Access denied. Check Azure RBAC permissions."),
        ("InvalidResourceName",       "Invalid container or blob name format."),
        ("AccountIsDisabled",         "Azure storage account is disabled."),
    ],
}


# ── Credential Pre-Validation ─────────────────────────────────────────────────

def validate_credentials(
    platform: str,
    credentials: Dict[str, Any],
    container: str = "",
) -> Tuple[bool, List[str]]:
    """
    Checks required credential fields locally before attempting a push.

    Args:
        platform:    "AWS S3", "GCS", or "Azure Blob"
        credentials: dict of credential key-value pairs
        container:   bucket or container name

    Returns:
        (valid, missing_fields_list)
        If valid = True, missing_fields_list is empty.
    """
    required = _REQUIRED_FIELDS.get(platform, [])
    missing = []

    for field in required:
        val = credentials.get(field, "")
        if not val or (isinstance(val, str) and not val.strip()):
            missing.append(field)

    if not container or not container.strip():
        missing.append("bucket/container name")

    return (len(missing) == 0, missing)


# ── Error Mapping ─────────────────────────────────────────────────────────────

def map_push_error(platform: str, error_string: str) -> str:
    """
    Maps raw SDK error strings to human-friendly messages.

    Args:
        platform:     "AWS S3", "GCS", or "Azure Blob"
        error_string: the raw error message or exception string

    Returns:
        Friendly error message, or a generic fallback if no pattern matches.
    """
    err_lower = str(error_string).lower()

    for pattern, friendly_msg in _ERROR_MAP.get(platform, []):
        if pattern.lower() in err_lower:
            return friendly_msg

    return f"Connection failed ({platform}). Please verify your credentials and try again."


# ── Push Filename Generation ──────────────────────────────────────────────────

def generate_push_filename(
    doc_id: str,
    export_format: str,
    timestamp: datetime.datetime | None = None
) -> str:
    """
    Generates a timestamped filename for cloud push.

    Examples:
        generate_push_filename("invoice_13442", "json")
        → "invoice_13442__20260301T2237.json"

        generate_push_filename("resume_john", "csv")
        → "resume_john__20260301T2237.csv"
    """
    ts = timestamp or datetime.datetime.now()
    ts_str = ts.strftime("%Y%m%dT%H%M")

    # Sanitize doc_id: replace unsafe chars with underscores
    safe_id = re.sub(r"[^\w\-.]", "_", str(doc_id).strip())
    if not safe_id:
        safe_id = "aurora_export"

    ext_map = {
        "JSON": ".json",
        "CSV": ".csv",
        "EXCEL": ".xlsx",
        "json": ".json",
        "csv": ".csv",
        "excel": ".xlsx",
        "xlsx": ".xlsx",
    }
    ext = ext_map.get(export_format, f".{export_format.lower()}")

    return f"{safe_id}__{ts_str}{ext}"


# ── Push Verification Helper ──────────────────────────────────────────────────

def build_verification_summary(
    platform: str,
    destination: str,
    local_size: int,
    remote_size: int = -1,
    checksum_match: bool | None = None,
) -> Tuple[bool, str]:
    """
    Builds a post-push verification summary.

    Returns:
        (verified, message)
    """
    checks = []

    # Size check
    if remote_size >= 0:
        if remote_size > 0:
            checks.append(f"Remote size: {remote_size:,} bytes")
            if remote_size == local_size:
                checks.append("Size match: exact")
            else:
                checks.append(f"Size mismatch: local={local_size:,}, remote={remote_size:,}")
        else:
            return False, f"Push verification failed: remote object at {destination} has size 0."
    else:
        checks.append("Remote size: not checked")

    # Checksum check
    if checksum_match is True:
        checks.append("Checksum: verified")
    elif checksum_match is False:
        return False, f"Push verification failed: checksum mismatch at {destination}."

    summary = f"Push verified to {platform} → {destination}. " + " | ".join(checks)
    return True, summary
