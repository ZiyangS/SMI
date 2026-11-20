"""Create Semantic IDs and aggregate frozen node embeddings along their paths."""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hierarchy import HIERARCHY_FILES


@dataclass(frozen=True)
class Level:
    name: str
    code: str
    description: str
    order: str


LEVELS = {
    "diag": (
        Level("Major", "CCSParentCategory", "CCSParentString", "numeric"),
        Level("Sub", "CCSChildCategory", "CCSChildString", "numeric"),
        Level("CCS", "CCSCode", "CCSString", "numeric"),
        Level("ICDIntegerCategory", "ICDIntegerCategory", "ICDIntegerString", "lexical"),
        Level("ICD", "ICD", "ICDString", "lexical"),
    ),
    "proc": (
        Level("Proc_L1", "L1_Code", "L1_Label", "numeric"),
        Level("Proc_L2", "L2_Code", "L2_Label", "numeric"),
        Level("Proc_L3", "L3_Code", "L3_Label", "length"),
        Level("Proc_L4", "L4_Code", "L4_Label", "length"),
        Level("Proc", "Procedure_Code", "Procedure_Description", "lexical"),
    ),
    "atc": (
        Level("ATC_L1", "ATC_L1_Code", "ATC_L1_Name", "atc"),
        Level("ATC_L2", "ATC_L2_Code", "ATC_L2_Name", "atc"),
        Level("ATC_L3", "ATC_L3_Code", "ATC_L3_Name", "atc"),
        Level("ATC_L4", "ATC_L4_Code", "ATC_L4_Name", "atc"),
        Level("ATC", "ATC_Code", "ATC_Name", "atc"),
    ),
}
METADATA = ["GlobalID", "Layer", "LayerTokenID", "Code", "String"]
NUMERIC_CODE = re.compile(r"\d+(?:\.\d+)*\Z")
ATC_CODE = re.compile(r"[A-Z]\d{2}[A-Z]?[A-Z]?(?:\d{2})?\Z")
VECTOR_COLUMNS = [str(index) for index in range(768)]
CODEBOOK_COLUMNS = METADATA + VECTOR_COLUMNS
BATCH_ROWS = 4096
DIAG_EVAL_CODEBOOK = "codebook_hier_w_embeddings_diag_SMI.csv"
DOMAIN_NAMES = {"diag": "diagnosis", "proc": "procedure", "atc": "medication"}


# Semantic IDs

def code_sort_key(code: str, order: str) -> tuple:
    code = code.strip()
    if order == "numeric":
        return tuple(map(int, code.split("."))) if NUMERIC_CODE.fullmatch(code) else (10**9, code)
    if order == "length":
        return len(code), code
    # Within an ATC level, the fixed-width codes have the legacy prefix order.
    upper = code.upper()
    return not bool(ATC_CODE.fullmatch(upper)), upper


def build_codebook(domain: str, hierarchy: pd.DataFrame) -> pd.DataFrame:
    tables = []
    for level in LEVELS[domain]:
        table = hierarchy[[level.code, level.description]].copy()
        table.columns = ["Code", "String"]
        described = table["String"].str.strip().ne("")
        if domain == "diag":
            table = table.loc[described].drop_duplicates(["Code", "String"], keep="first")
        else:
            table["Code"] = table["Code"].str.strip()
            table = table.loc[described & table["Code"].ne("")].drop_duplicates("Code", keep="first")
        if level.order == "lexical":
            table = table.sort_values(["Code", "String"], kind="stable")
        else:
            table = table.sort_values("Code", kind="stable", key=lambda codes: codes.map(
                lambda code: code_sort_key(code, level.order)
            ))
        table["Layer"] = level.name
        table["LayerTokenID"] = range(len(table))
        tables.append(table)
    codebook = pd.concat(tables, ignore_index=True)
    codebook["GlobalID"] = range(len(codebook))
    return codebook[METADATA]


def semantic_ids(domain: str, hierarchy: pd.DataFrame, codebook: pd.DataFrame, full_procedure_paths: bool = False):
    levels = LEVELS[domain]
    normalized = hierarchy
    if domain in ("proc", "atc"):
        normalized = hierarchy.apply(lambda column: column.str.strip())
    tokens = {
        (layer, code): str(token)
        for layer, code, token in codebook[["Layer", "Code", "LayerTokenID"]].itertuples(index=False, name=None)
    }
    columns = [column for level in levels for column in (level.code, level.description)]
    paths = [dict() for _ in levels]
    for row in normalized[columns].itertuples(index=False, name=None):
        parts = []
        for depth, level in enumerate(levels):
            code, description = row[depth * 2:depth * 2 + 2]
            token = tokens.get((level.name, code), "") if description.strip().lower() not in ("", "nan") else " "
            parts.append(token)
            paths[depth].setdefault(code, ".".join(parts))
    for level, path in zip(levels, paths):
        ids = codebook.loc[codebook["Layer"].eq(level.name), ["Code"]].copy()
        ids["SemanticID"] = ids["Code"].map(path).fillna("")
        if domain == "proc" and not full_procedure_paths:
            # Historical procedure IDs had an empty self slot at every depth.
            ids["SemanticID"] = ids["SemanticID"].str.replace(r"[^.]*$", "", regex=True)
        yield ids


# Embedding generation

def embedding_path(output: Path, domain: str, depth: int) -> Path:
    basename = f"{DOMAIN_NAMES[domain]}_SMI_embeddings"
    if depth == len(LEVELS[domain]):
        return output / f"{basename}.csv"
    return output / "smi_embeddings_by_level" / f"{basename}_L{depth}.csv"


def validate_node_header(source: Path) -> None:
    if pd.read_csv(source, nrows=0).columns.tolist() != CODEBOOK_COLUMNS:
        raise ValueError(f"Expected the legacy codebook header with 768 vector columns in {source}")


def read_node_vectors(domain: str, source: Path, reference: pd.DataFrame) -> dict[str, np.ndarray]:
    validate_node_header(source)
    # ATC's original aggregation converted decimal text directly to Python float.
    # ICD and procedures used the default pandas numeric parser.
    dtype = str if domain == "atc" else dict.fromkeys(METADATA, str)
    keys = ["GlobalID", "Layer", "LayerTokenID", "Code"]
    vectors = np.empty((len(reference), len(VECTOR_COLUMNS)), dtype=np.float64)
    offset = 0
    for chunk in pd.read_csv(source, dtype=dtype, chunksize=BATCH_ROWS):
        end = offset + len(chunk)
        expected = reference.iloc[offset:end][keys].astype(str).reset_index(drop=True)
        actual = chunk[keys].fillna("").reset_index(drop=True)
        if not actual.equals(expected):
            raise ValueError(f"Codebook rows {offset + 1}–{end} in {source} do not match the generated reference")
        vectors[offset:end] = chunk[VECTOR_COLUMNS].to_numpy(dtype=np.float64)
        offset = end
    if offset != len(reference):
        raise ValueError(f"Codebook has {offset} rows, expected {len(reference)} in {source}")
    positions = reference.groupby("Layer", sort=False).indices
    return {layer: vectors[indices[0]:indices[-1] + 1] for layer, indices in positions.items()}


def pooled_batches(domain: str, ids: pd.DataFrame, node_vectors: dict[str, np.ndarray]):
    # Semantic IDs use token indices from the same codebook as node_vectors.
    for start in range(0, max(len(ids), 1), BATCH_ROWS):
        batch = ids.iloc[start:start + BATCH_ROWS].copy()
        if domain in ("proc", "atc"):
            batch["SemanticID"] = batch["SemanticID"].str.strip()
        tokens = batch["SemanticID"].str.split(".", regex=False, expand=True).fillna("")
        pooled = np.zeros((len(batch), len(VECTOR_COLUMNS)), dtype=np.float64)
        for position in tokens.columns:
            level = LEVELS[domain][position].name
            column = tokens[position]
            populated = column.str.strip().ne("").to_numpy()
            if not populated.any():
                continue
            indices = column.loc[populated].to_numpy(dtype=np.int64)
            pooled[populated] += node_vectors[level][indices]
        yield pd.concat([batch.reset_index(drop=True), pd.DataFrame(pooled, columns=VECTOR_COLUMNS)], axis=1)


def aggregate(domain: str, source: Path, tables: Path, write_csv, *, full_procedure_paths: bool = False) -> None:
    hierarchy = pd.read_csv(tables / HIERARCHY_FILES[domain], dtype=str).fillna("")
    reference = build_codebook(domain, hierarchy)
    vectors = read_node_vectors(domain, source, reference)
    ids_by_level = semantic_ids(domain, hierarchy, reference, full_procedure_paths)
    for depth, ids in enumerate(ids_by_level, start=1):
        write_csv(embedding_path(tables, domain, depth), pooled_batches(domain, ids, vectors))


def float32_vectors(frame: pd.DataFrame) -> np.ndarray:
    # Match the original exporter: to_numeric on CSV strings, then downcast.
    return frame[VECTOR_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)


def diagnosis_codebook_batches(source: Path, tables: Path):
    validate_node_header(source)
    pooled = {}
    for depth, level in enumerate(LEVELS["diag"], start=1):
        path = embedding_path(tables, "diag", depth)
        codes, arrays = [], []
        for chunk in pd.read_csv(path, dtype=str, chunksize=BATCH_ROWS):
            codes.extend(chunk["Code"].fillna(""))
            arrays.append(float32_vectors(chunk))
        code_index = pd.Index(codes)
        first = ~code_index.duplicated(keep="first")
        data = np.concatenate(arrays)[first]
        pooled[level.name] = (code_index[first].str.strip(), data)

    for chunk in pd.read_csv(source, dtype=str, chunksize=BATCH_ROWS):
        chunk = chunk.fillna("").reset_index(drop=True)
        metadata = chunk[METADATA].copy()
        vectors = float32_vectors(chunk)
        for layer, (code_index, data) in pooled.items():
            rows = np.flatnonzero(metadata["Layer"].str.strip().eq(layer).to_numpy())
            indices = code_index.get_indexer(metadata.loc[rows, "Code"].str.strip())
            matched = indices >= 0
            replacements = data[indices[matched]]
            available = ~np.isnan(replacements).all(axis=1)
            vectors[rows[matched][available]] = replacements[available]
        yield pd.concat([metadata, pd.DataFrame(vectors, columns=VECTOR_COLUMNS)], axis=1)
