"""Redact one historical machine locator without changing historical metrics."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/"audit/legacy_r2/validity_ablation_results.json"
original_bytes = path.read_bytes()
content = json.loads(original_bytes.decode("utf-8"))
record = content["source_batch_summary"]["excluded_identities"][0]
locator = record["source_locator"]
if re.search(r"[A-Za-z]:[\\/]+Users[\\/]+", locator):
    normalized = re.sub(r"[\\/]+", "/", locator)
    assert "/outputs/" in normalized
    record["source_locator"] = "${ARCHIVE_ROOT}/outputs/" + normalized.split("/outputs/", 1)[1]
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    audit = {"file": "audit/legacy_r2/validity_ablation_results.json", "original_sha256": hashlib.sha256(original_bytes).hexdigest(), "published_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "field": "source_batch_summary.excluded_identities[0].source_locator", "change": "Historical machine-specific absolute locator replaced by an explicit ${ARCHIVE_ROOT}/outputs relative locator. No metric, count, prediction or source filename changed."}
    (ROOT/"audit/legacy_locator_normalisation.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("Normalised the historical locator; numerical JSON values unchanged.")
else:
    print("Historical locator already uses a portable archive token.")
