"""Build five-level diagnosis, procedure, and medication hierarchies."""

import csv
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import pandas as pd


HIERARCHY_FILES = {
    "diag": "diagnosis_hierarchy.csv",
    "proc": "procedure_hierarchy.csv",
    "atc": "medication_hierarchy.csv",
}

DIAG_COLUMNS = [
    "ICD", "Flag", "ICDString", "CCSCode", "CCSString",
    "CCSParentCategory", "CCSParentString", "CCSChildCategory", "CCSChildString",
    "ICDIntegerCategory", "ICDIntegerString",
]
PROC_COLUMNS = [
    "Procedure_Code", "Procedure_Description", "L4_Code", "L4_Label",
    "L3_Code", "L3_Label", "L2_Code", "L2_Label", "L1_Code", "L1_Label",
]


# Diagnosis hierarchy

def icd9_ccs(source: Path) -> dict[str, tuple[str, str]]:
    mapping = {}
    category = None
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            if not line[0].isspace():
                header = line.strip().split(maxsplit=1)
                if len(header) == 2 and header[0].isdigit():
                    category = tuple(header)
                continue
            if category is not None:
                for code in line.split():
                    mapping.setdefault(code, category)
    return mapping


def match_icd9(codes: pd.Series, source: Path) -> list[tuple[str, str]]:
    exact = icd9_ccs(source)
    prefix = {}
    for code, category in exact.items():
        prefix.setdefault(code, category)
        for length in range(1, len(code)):
            if code[length:].isdigit():
                prefix.setdefault(code[:length], category)

    matches = []
    for code in codes.str.strip():
        match = exact.get(code + "00") or exact.get(code + "0") or prefix.get(code)
        if match is None and "." in code:
            compact = code.replace(".", "")
            if len(compact) in (4, 5):
                match = exact.get(compact)
        matches.append(match or ("", ""))
    return matches


def match_icd10(codes: pd.Series, source: Path) -> list[tuple[str, str]]:
    exact = {}
    first_variant = {}
    zero_variant = {}
    with (source / "ICD10CM_to_CCSCM.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            code = row["ICD10CM"].strip()
            exact[code] = row["CCSCM"].strip()
            prefix = code.split(".", 1)[0]
            first_variant.setdefault(prefix, code)
            if code.endswith((".0", ".00")):
                zero_variant.setdefault(prefix, code)
    with (source / "CCSCM.csv").open(encoding="utf-8", newline="") as stream:
        labels = {row["code"].strip(): row["name"].strip() for row in csv.DictReader(stream)}

    matches = []
    for code in codes.str.strip():
        if code in exact or "." in code:
            category = exact.get(code, "")
        else:
            category = exact.get(zero_variant.get(code), "") or exact.get(first_variant.get(code), "")
        matches.append((category, labels.get(category, "")))
    return matches


def ccs_hierarchy(source: Path) -> dict[str, tuple[str, str, str, str]]:
    labels_path = source / "ccs_multilevel_code_string.csv"
    labels = pd.read_csv(labels_path, dtype=str, keep_default_na=False)
    titles = dict(zip(labels["Multi_CCSCode"].str.strip(), labels["Multi_CCSString"].str.strip()))
    assignments = defaultdict(lambda: ["", "", "", ""])
    links = pd.read_csv(source / "ccs_multilevel_to_single.csv", dtype=str)
    for multi, single in links.itertuples(index=False, name=None):
        section = 2 if "." in multi else 0
        assignments[single][section:section + 2] = [multi, titles.get(multi, "")]
    return {code: tuple(path) for code, path in assignments.items()}


def build_diagnosis_versions(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(source / "Phecode_map_v1_2_icd9_icd10cm.csv", dtype=str)
    paths = ccs_hierarchy(source)
    higher_levels = ["CCSParentCategory", "CCSParentString", "CCSChildCategory", "CCSChildString"]
    versions = []
    for version in ("9", "10"):
        codes = labels.loc[labels["Flag"].str.strip().eq(version), ["ICD", "Flag", "ICDString"]].reset_index(drop=True)
        if version == "9":
            matches = match_icd9(codes["ICD"], source / "AppendixASingleDX.txt")
        else:
            matches = match_icd10(codes["ICD"], source)
        codes[["CCSCode", "CCSString"]] = pd.DataFrame(matches, index=codes.index)
        codes = codes.loc[codes["CCSCode"].ne("") & codes["CCSString"].ne("")].copy()
        codes[higher_levels] = pd.DataFrame(
            [paths.get(category, ("", "", "", "")) for category in codes["CCSCode"]],
            index=codes.index,
        )
        codes["ICD"] = codes["ICD"].str.strip()
        codes["ICDString"] = codes["ICDString"].str.strip()
        codes["ICDIntegerCategory"] = codes["ICD"].str.split(".", n=1, regex=False).str[0]
        integers = codes.loc[codes["ICD"].eq(codes["ICDIntegerCategory"])].drop_duplicates("ICD", keep="first")
        codes["ICDIntegerString"] = codes["ICDIntegerCategory"].map(integers.set_index("ICD")["ICDString"])
        versions.append(codes)
    return versions[0], versions[1]


def build_diagnosis(versions: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(versions, ignore_index=True)[DIAG_COLUMNS]
    hierarchy = combined.drop_duplicates("ICD", keep="first").copy()
    has_description = hierarchy["ICDIntegerString"].fillna("").str.strip().ne("")
    counts = hierarchy.loc[has_description].groupby("ICDIntegerCategory")["CCSCode"].transform("nunique")
    split_prefix = has_description & counts.reindex(hierarchy.index).gt(1)
    hierarchy.loc[split_prefix, "ICDIntegerCategory"] = (
        hierarchy.loc[split_prefix, "ICDIntegerCategory"].astype(str)
        + "@"
        + hierarchy.loc[split_prefix, "CCSCode"].astype(str)
    )
    return hierarchy


# Procedure hierarchy

SINGLE_CATEGORY = re.compile(r"^(\d+)\s+(.+)$")
MULTI_CATEGORY = re.compile(r"^(\d+(?:\.\d+)?)\s{2,}(\S.*)")
ICD9_CODE = re.compile(r"\d{3,4}\Z")


def read_labels(path: Path) -> dict[str, str]:
    labels = pd.read_csv(path, dtype=str, keep_default_na=False)
    return dict(labels.itertuples(index=False, name=None))


def procedure_categories(source: Path) -> tuple[dict[str, str], dict[str, str]]:
    by_code, labels = {}, {}
    category = None
    for line in source.read_text(encoding="latin-1").splitlines():
        if header := SINGLE_CATEGORY.match(line):
            category, label = header.groups()
            labels[category] = label
        elif category and line[:1].isspace():
            for code in line.split():
                if ICD9_CODE.fullmatch(code):
                    by_code.setdefault(code, category)
    return by_code, labels


def procedure_parents(source: Path) -> tuple[dict[str, str], dict[str, str]]:
    parents, labels = {}, {}
    for line in source.read_text(encoding="latin-1").splitlines():
        header = MULTI_CATEGORY.match(line)
        if header is None:
            continue
        code, payload = header.groups()
        label = re.split(r"\s{2,}", payload.strip(), maxsplit=1)[0]
        plain, _, references = label.partition("[")
        labels[code] = plain.rstrip()
        if "." in code:
            for single in re.findall(r"(?<!\d)(\d+)\.", references):
                parents[single] = code
    return parents, labels


def build_icd9_procedures(source: Path, prefixes: pd.Series) -> pd.DataFrame:
    categories, category_labels = procedure_categories(source / "AppendixBSinglePR.txt")
    parents, parent_labels = procedure_parents(source / "AppendixDMultiPR.txt")
    descriptions = read_labels(source / "icd9_procedure_descriptions.csv")
    category_labels.update(read_labels(source / "icd9_ccs_label_overrides.csv"))
    parents.update(read_labels(source / "icd9_ccs_parent_overrides.csv"))
    unknown = descriptions.keys() - categories.keys()
    if unknown:
        raise ValueError(f"ICD-9 procedures absent from Appendix B: {sorted(unknown)[:10]}")

    frame = pd.DataFrame(
        ((code, descriptions[code], category) for code, category in categories.items() if code in descriptions),
        columns=["Procedure_Code", "Procedure_Description", "L3_Code"],
    )
    frame["L4_Code"] = frame["Procedure_Code"].str[:3]
    frame["L4_Label"] = frame["L4_Code"].map(prefixes).fillna("Category " + frame["L4_Code"])
    frame["L3_Label"] = frame["L3_Code"].map(category_labels)
    frame["L2_Code"] = frame["L3_Code"].map(parents)
    frame["L1_Code"] = frame["L2_Code"].str.split(".", n=1, regex=False).str[0]
    for level in (1, 2):
        frame[f"L{level}_Label"] = frame[f"L{level}_Code"].map(parent_labels)
    if frame[PROC_COLUMNS].isna().any().any():
        raise ValueError("Incomplete parent or label mapping in the historical ICD-9 procedure inputs")
    return frame[PROC_COLUMNS]


def build_icd10_procedures(source: Path, prefixes: pd.Series) -> pd.DataFrame:
    archive = source / "ccs_pr_icd10pcs_2020_1.zip"
    with ZipFile(archive) as z, z.open("ccs_pr_icd10pcs_2020_1.csv") as stream:
        frame = pd.read_csv(stream, dtype=str, keep_default_na=False)
    frame.columns = [
        "Procedure_Code", "L3_Code", "Procedure_Description", "L3_Label",
        "L1_Code", "L1_Label", "L2_Code", "L2_Label",
    ]
    frame = frame.apply(lambda column: column.str.strip().str.strip("'").str.strip())
    frame["L4_Code"] = frame["Procedure_Code"].str[:3]
    frame["L4_Label"] = frame["L4_Code"].map(prefixes)
    if frame["L4_Label"].isna().any():
        raise ValueError("Missing prefix label in the historical ICD-10-PCS procedure inputs")
    return frame[PROC_COLUMNS]


def build_procedure_versions(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefixes = pd.read_csv(source / "procedure_prefix_labels.csv", dtype=str, keep_default_na=False)
    prefixes = prefixes.set_index(["Version", "L4_Code"])["L4_Label"]
    return build_icd9_procedures(source, prefixes.loc["9"]), build_icd10_procedures(source, prefixes.loc["10"])


def build_procedure(versions: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    labeled = [frame.assign(ProcSystem=system) for frame, system in zip(versions, ("ICD9PROC", "ICD10PCS"))]
    combined = pd.concat(labeled, ignore_index=True)
    return combined.fillna("").apply(lambda column: column.str.strip()).drop_duplicates()


# Medication hierarchy

def build_atc(source: Path) -> pd.DataFrame:
    catalog = pd.read_csv(source, usecols=["atc_code", "atc_name"], dtype=str)
    catalog = catalog.astype(str).apply(lambda column: column.str.strip())
    titles = dict(zip(catalog["atc_code"], catalog["atc_name"]))
    hierarchy = catalog.loc[catalog["atc_code"].str.len().eq(7), ["atc_code", "atc_name"]].copy()
    if hierarchy.empty:
        raise ValueError(f"No seven-character ATC codes in {source}")
    hierarchy.columns = ["ATC_Code", "ATC_Name"]
    for level, width in ((4, 5), (3, 4), (2, 3), (1, 1)):
        prefixes = hierarchy["ATC_Code"].str[:width]
        hierarchy[f"ATC_L{level}_Code"] = prefixes
        hierarchy[f"ATC_L{level}_Name"] = prefixes.map(titles).fillna("Unknown")
    return hierarchy
