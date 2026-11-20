import argparse
import tempfile
from pathlib import Path

from embeddings import DIAG_EVAL_CODEBOOK, LEVELS, aggregate, diagnosis_codebook_batches
from hierarchy import (
    HIERARCHY_FILES,
    build_atc,
    build_diagnosis,
    build_diagnosis_versions,
    build_procedure,
    build_procedure_versions,
)


ROOT = Path(__file__).resolve().parent
NODE_FILES = {
    "diag": "codebook_hier_w_embeddings_diag.csv",
    "proc": "codebook_hier_w_embeddings_proc.csv",
    "atc": "codebook_atc_hier_w_embeddings.csv",
}


def write_csv(destination: Path, batches) -> None:
    destination = destination.resolve()
    if not destination.is_relative_to(ROOT):
        raise ValueError(f"Outputs must be inside {ROOT}: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".smi-", dir=destination.parent) as folder:
        temporary = Path(folder) / destination.name
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            for index, frame in enumerate(batches):
                frame.to_csv(stream, index=False, header=index == 0, lineterminator="\n")
        temporary.replace(destination)
    print(f"Wrote {destination.relative_to(ROOT)}", flush=True)


def make_hierarchies(inputs: Path, output: Path, domains: tuple[str, ...]) -> None:
    if "diag" in domains:
        versions = build_diagnosis_versions(inputs / "diagnosis")
        hierarchy = build_diagnosis(versions)
        write_csv(output / HIERARCHY_FILES["diag"], (hierarchy,))
    if "proc" in domains:
        versions = build_procedure_versions(inputs / "procedure")
        hierarchy = build_procedure(versions)
        write_csv(output / HIERARCHY_FILES["proc"], (hierarchy,))
    if "atc" in domains:
        hierarchy = build_atc(inputs / "medication" / "WHO ATC-DDD 2024-07-31.csv")
        write_csv(output / HIERARCHY_FILES["atc"], (hierarchy,))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SMI hierarchies, Semantic IDs and embedding CSVs.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    stages = parser.add_subparsers(dest="stage", required=True)
    hierarchy = stages.add_parser("build-hierarchy")
    hierarchy.add_argument("--inputs", type=Path, default=ROOT / "inputs")
    hierarchy.add_argument("--domain", choices=tuple(LEVELS) + ("all",), default="all")
    all_stages = stages.add_parser("reproduce", help="Build all three domains from the historical inputs")
    all_stages.add_argument("--inputs", type=Path, default=ROOT / "inputs")
    all_stages.add_argument("--node-codebooks", type=Path, required=True)
    pooling = stages.add_parser("aggregate")
    pooling.add_argument("domain", choices=LEVELS)
    pooling.add_argument("--full-procedure-paths", action="store_true", help="Include the procedure's current-layer token")
    export = stages.add_parser("export-eval-codebook", help="Export the diagnosis codebook with CCS suffixes")
    export.add_argument("domain", choices=["diag"])
    for command in (pooling, export):
        command.add_argument("--node-codebook", type=Path, required=True)

    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT) or output == ROOT:
        parser.error("--output-dir must be a subdirectory of SMI_release")
    if args.stage == "reproduce":
        sources = {domain: args.node_codebooks / name for domain, name in NODE_FILES.items()}
        for source in sources.values():
            if not source.is_file():
                parser.error(f"Missing node codebook: {source}")
        make_hierarchies(args.inputs, output, tuple(LEVELS))
        for domain, source in sources.items():
            aggregate(domain, source, output, write_csv)
        return

    if args.stage == "build-hierarchy":
        domains = tuple(LEVELS) if args.domain == "all" else (args.domain,)
        make_hierarchies(args.inputs, output, domains)
    elif args.stage == "aggregate":
        if args.full_procedure_paths and args.domain != "proc":
            pooling.error("--full-procedure-paths requires the proc domain")
        aggregate(args.domain, args.node_codebook, output, write_csv, full_procedure_paths=args.full_procedure_paths)
    else:
        write_csv(output / DIAG_EVAL_CODEBOOK, diagnosis_codebook_batches(args.node_codebook, output))


if __name__ == "__main__":
    main()
