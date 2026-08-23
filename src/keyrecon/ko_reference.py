from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class KoreanReference:
    root: Path
    payload: dict

    @property
    def profile(self) -> dict:
        return self.payload["profile"]

    @property
    def weights(self) -> dict[str, float]:
        return {
            str(k): float(v)
            for k, v in self.profile["weights"].items()
        }

    @property
    def threshold(self) -> float:
        return float(self.profile["minimum_cks_score"])

    @property
    def score_decimals(self) -> int:
        return int(self.profile["decision_score_round_decimals"])

    @property
    def top_k(self) -> int:
        return int(self.profile["top_k"])

    @property
    def configuration_id(self) -> str:
        return str(self.profile["configuration_id"])

    def asset_path(self, *keys: str) -> Path:
        node = self.payload
        for key in keys:
            node = node[key]
        file_name = node["file"]
        return self.root / file_name

    def verify_assets(self) -> list[dict]:
        asset_nodes = [
            ("candidate_rule_contract", self.payload["candidate_generation"]["rule_contract_asset"]),
            ("full_development_edge_inventory", self.payload["candidate_generation"]["edge_inventory_asset"]),
            ("canonicalization_contract", self.payload["canonicalization"]["contract_asset"]),
            ("feature_definition_contract", self.payload["components"]["feature_definition_contract_asset"]),
            ("tfidf_df_dispersion", self.payload["components"]["runtime_resources"]["tfidf_df_dispersion"]),
            ("domain_focus", self.payload["components"]["runtime_resources"]["domain_focus"]),
            ("phrase_quality", self.payload["components"]["runtime_resources"]["phrase_quality"]),
        ]

        rows = []
        for role, node in asset_nodes:
            p = self.root / node["file"]
            exists = p.exists()
            observed_size = p.stat().st_size if exists else None
            observed_sha = _sha256_file(p) if exists else None
            ok = (
                exists
                and observed_size == int(node["size_bytes"])
                and observed_sha == str(node["sha256"])
            )
            rows.append(
                {
                    "asset_role": role,
                    "file_name": node["file"],
                    "exists": exists,
                    "expected_size_bytes": int(node["size_bytes"]),
                    "observed_size_bytes": observed_size,
                    "expected_sha256": str(node["sha256"]),
                    "observed_sha256": observed_sha,
                    "status": "PASS" if ok else "FAIL",
                }
            )
        return rows


def load_korean_reference(
    reference_root: str | Path | None = None,
) -> KoreanReference:
    if reference_root is None:
        reference_root = (
            Path(__file__).resolve().parent
            / "resources"
            / "ko_reference_v1"
        )
    else:
        reference_root = Path(reference_root)

    reference_path = reference_root / "K18_ko_reference_v1_00.json"

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Korean reference JSON is missing: {reference_path}"
        )

    payload = json.loads(
        reference_path.read_text(encoding="utf-8-sig")
    )

    required = {
        "schema_name": "keyrecon.ko_reference",
        "schema_version": "1.0",
        "reference_id": "ko_reference_v1",
        "language": "ko",
        "authoritative": True,
        "status": "FROZEN",
    }

    for key, expected in required.items():
        observed = payload.get(key)
        if observed != expected:
            raise RuntimeError(
                f"Korean reference contract drift: "
                f"{key}={observed!r}, expected={expected!r}"
            )

    if payload["profile"]["configuration_id"] != "W034_S45":
        raise RuntimeError("Unexpected Korean reference configuration.")

    if float(payload["profile"]["minimum_cks_score"]) != 0.45:
        raise RuntimeError("Unexpected Korean reference threshold.")

    if int(payload["profile"]["decision_score_round_decimals"]) != 12:
        raise RuntimeError("Unexpected Korean reference score rounding.")

    if int(payload["profile"]["top_k"]) != 10:
        raise RuntimeError("Unexpected Korean reference top-k.")

    ref = KoreanReference(
        root=reference_root,
        payload=payload,
    )

    audit = ref.verify_assets()
    failed = [row for row in audit if row["status"] != "PASS"]

    if failed:
        raise RuntimeError(
            "Korean reference asset verification failed: "
            + repr(failed)
        )

    return ref
