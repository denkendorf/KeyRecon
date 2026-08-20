from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

from .engine import KeyRecon
from .languages.registry import LANGUAGE_SPECS
from .version import __version__

def _models(_args) -> int:
    print("code\tlanguage\tstatus\tspaCy model")
    for code, spec in LANGUAGE_SPECS.items():
        print(f"{code}\t{spec.label}\t{spec.status}\t{spec.model}")
    return 0

def _setup(args) -> int:
    spec = LANGUAGE_SPECS[args.lang]
    print(f"KeyRecon {__version__}: {spec.label} ({spec.status})")
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
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0

def _run(args) -> int:
    inp = Path(args.input)
    out = Path(args.output)
    df = pd.read_csv(inp, encoding="utf-8-sig", keep_default_na=False)
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
    pred.to_csv(out, index=False, encoding="utf-8-sig")
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(engine.manifest(), ensure_ascii=False, indent=2),
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
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("models", help="List language profiles.")
    p.set_defaults(func=_models)

    p = sub.add_parser("setup", help="Show the spaCy model installation command.")
    p.add_argument("--lang", choices=LANGUAGE_SPECS, required=True)
    p.set_defaults(func=_setup)

    p = sub.add_parser("profile", help="Inspect a language profile.")
    p.add_argument("--lang", choices=LANGUAGE_SPECS, required=True)
    p.set_defaults(func=_profile)

    p = sub.add_parser("run", help="Fit on observed author keywords and reconstruct missing fields.")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--lang", choices=LANGUAGE_SPECS, default="en")
    p.add_argument("--mode", choices=["adaptive", "reference"], default="adaptive")
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
