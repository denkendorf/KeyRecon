from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .engine import KeyRecon
from .ko_reference import load_korean_reference
from .ko_runtime import run_korean_adaptive
from .languages.registry import LANGUAGE_SPECS
from .version import __version__


def _models(_args) -> int:
    print("code\tlanguage\tstatus\truntime/model")
    for code, spec in LANGUAGE_SPECS.items():
        runtime = (
            f"{spec.model}=={spec.expected_model_version}"
            if code == "ko" and spec.expected_model_version
            else spec.model
        )
        print(f"{code}\t{spec.label}\t{spec.status}\t{runtime}")
    return 0


def _setup(args) -> int:
    spec = LANGUAGE_SPECS[args.lang]

    print(f"KeyRecon {__version__}: {spec.label} ({spec.status})")

    if args.lang == "ko":
        print("Install the authoritative Korean runtime with:")
        print("  python -m pip install 'keyrecon[korean]'")
        print("Required runtime:")
        print("  kiwipiepy==0.23.2")
        return 0

    print("Install the selected spaCy pipeline explicitly:")
    print(f"  python -m spacy download {spec.model}")

    if "trf" in spec.model:
        print("Transformer profiles also require spaCy transformer dependencies:")
        print("  python -m pip install 'spacy-transformers>=1.3,<2'")

    return 0


def _profile(args) -> int:
    spec = LANGUAGE_SPECS[args.lang]

    payload = {
        "language": spec.code,
        "label": spec.label,
        "status": spec.status,
        "model": spec.model,
        "candidate_profile": spec.candidate_profile,
        "expected_model_version": spec.expected_model_version,
        "reference_spacy_version": spec.reference_spacy_version,
    }

    if args.lang == "ko":
        ref = load_korean_reference()

        payload["reference_id"] = ref.payload["reference_id"]
        payload["configuration_id"] = ref.configuration_id
        payload["minimum_cks_score"] = ref.threshold
        payload["decision_score_round_decimals"] = ref.score_decimals
        payload["top_k"] = ref.top_k
        payload["weights"] = ref.weights
        payload["fit_performed_at_inference"] = False
        payload["authoritative_extensions_enabled"] = (
            ref.payload["post_core_extension_policy"][
                "authoritative_extensions_enabled"
            ]
        )

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run_korean(args, df: pd.DataFrame, out: Path) -> int:
    required = {
        args.id_col,
        args.title_col,
        args.abstract_col,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Input records lack {sorted(missing)}"
        )

    ref = load_korean_reference()

    if args.top_k != ref.top_k:
        raise ValueError(
            f"The frozen Korean reference requires --top-k {ref.top_k}; "
            f"observed {args.top_k}."
        )

    if args.mode not in {"adaptive", "reference"}:
        raise ValueError(
            "Korean mode must be 'adaptive' or 'reference'."
        )

    if args.keywords_col in df.columns:
        raw_keywords = (
            df[args.keywords_col]
            .fillna("")
            .astype(str)
        )
        target_mask = raw_keywords.str.strip().eq("")
    else:
        raw_keywords = pd.Series(
            [""] * len(df),
            index=df.index,
            dtype=object,
        )
        target_mask = pd.Series(
            [True] * len(df),
            index=df.index,
            dtype=bool,
        )

    runtime_input = pd.DataFrame(
        {
            "record_id": df[args.id_col].astype(str),
            "title": df[args.title_col],
            "abstract": df[args.abstract_col],
            "DE_original": raw_keywords,
        }
    )

    targets = runtime_input.loc[target_mask].copy()

    result = run_korean_adaptive(targets)
    pred = result.predictions_long.copy()

    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(
        out,
        index=False,
        encoding="utf-8-sig",
    )

    manifest_path = out.with_suffix(
        out.suffix + ".manifest.json"
    )

    manifest = {
        "keyrecon_version": __version__,
        "language": "ko",
        "language_status": "reference",
        "reference_id": ref.payload["reference_id"],
        "configuration_id": ref.configuration_id,
        "requested_mode": args.mode,
        "execution_mode": "frozen_adaptive_reference",
        "fit_performed": False,
        "selected_weights": ref.weights,
        "selected_threshold": ref.threshold,
        "threshold_source": "frozen K18 ko_reference_v1",
        "decision_score_round_decimals": ref.score_decimals,
        "threshold_rule": ref.payload["profile"]["inclusive_threshold_rule"],
        "top_k": ref.top_k,
        "input_records": int(len(df)),
        "target_records": int(target_mask.sum()),
        "prediction_rows": int(len(pred)),
        "post_core_extensions_activated": False,
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"predictions: {out}")
    print(f"manifest:    {manifest_path}")
    print(f"reference:   {ref.payload['reference_id']}")
    print(f"config:      {ref.configuration_id}")
    print(f"threshold:   {ref.threshold:.2f}")
    print(f"weights:     {ref.weights}")
    print("fit:         False")
    return 0


def _run(args) -> int:
    inp = Path(args.input)
    out = Path(args.output)

    df = pd.read_csv(
        inp,
        encoding="utf-8-sig",
        keep_default_na=False,
    )

    if args.lang == "ko":
        return _run_korean(args, df, out)

    engine = KeyRecon(
        language=args.lang,
        mode=args.mode,
        n_folds=args.folds,
        fold_seed=args.seed,
        strict_reference=args.strict_reference,
    )

    pred = engine.fit_reconstruct_missing(
        df,
        id_col=args.id_col,
        title_col=args.title_col,
        abstract_col=args.abstract_col,
        keywords_col=args.keywords_col,
        delimiter=args.delimiter,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(
        out,
        index=False,
        encoding="utf-8-sig",
    )

    manifest_path = out.with_suffix(
        out.suffix + ".manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            engine.manifest(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"predictions: {out}")
    print(f"manifest:    {manifest_path}")
    print(f"threshold:   {engine.threshold_:.2f}")
    print(f"weights:     {engine.weights_.as_dict()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="keyrecon",
        description="Author-keyword reconstruction with Adaptive CKS.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    p = sub.add_parser(
        "models",
        help="List language profiles.",
    )
    p.set_defaults(func=_models)

    p = sub.add_parser(
        "setup",
        help="Show runtime/model installation guidance.",
    )
    p.add_argument(
        "--lang",
        choices=LANGUAGE_SPECS,
        required=True,
    )
    p.set_defaults(func=_setup)

    p = sub.add_parser(
        "profile",
        help="Inspect a language profile.",
    )
    p.add_argument(
        "--lang",
        choices=LANGUAGE_SPECS,
        required=True,
    )
    p.set_defaults(func=_profile)

    p = sub.add_parser(
        "run",
        help="Reconstruct missing author-keyword fields.",
    )
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument(
        "--lang",
        choices=LANGUAGE_SPECS,
        default="en",
    )
    p.add_argument(
        "--mode",
        choices=["adaptive", "reference"],
        default="adaptive",
        help=(
            "For Korean, both 'reference' and compatibility alias 'adaptive' "
            "execute frozen ko_reference_v1 without fitting."
        ),
    )
    p.add_argument("--id-col", default="record_id")
    p.add_argument("--title-col", default="title")
    p.add_argument("--abstract-col", default="abstract")
    p.add_argument("--keywords-col", default="author_keywords")
    p.add_argument("--delimiter", default=";")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=636)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--strict-reference", action="store_true")
    p.set_defaults(func=_run)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
