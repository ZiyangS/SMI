# SMI: Ontology-Grounded Semantic IDs for Hierarchy-Aware Medical Concept Representation

**NeurIPS 2026**

Ziyang Song, Lia Shen, Yixuan Li, Qincheng Lu, Weihao Li, Haoran Xu, He Zhu, Xinyu Wang, and Yue Li

## Overview

Biomedical language models can lose useful clinical context when encoding a rare or fine-grained medical code in isolation. **SMI (Semantic Medical Identifier)** uses the code's path through a clinical ontology to bring that context into its representation.

SMI assigns each concept a **Semantic ID**, an ordered path from a broad clinical category to the concept itself. For example, the ATC code `A01AA01` follows the path `A → A01 → A01A → A01AA → A01AA01`. SMI sums the frozen BioBERT node embeddings along this path to obtain a 768-dimensional concept embedding. The aggregation requires no training.

The paper evaluates these representations on clinical relation prediction within ICD and ATC and across the two ontologies, clinical prediction on MIMIC-IV, and cross-hospital distribution shift on eICU. This repository covers five-level hierarchy construction, Semantic ID generation, and the original unweighted SMI aggregation from precomputed node embeddings.

## Quick start

Run from the repository root. The reference outputs were generated with Python 3.13.5, NumPy 2.1.3, and pandas 2.2.3.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.1.3 pandas==2.2.3
```

The ontology inputs are in [`inputs/`](inputs/). Set `NODE_CODEBOOK_DIR` to a directory containing the three **unaggregated BioBERT node codebooks** below.

| Domain | File |
| --- | --- |
| Diagnosis | `codebook_hier_w_embeddings_diag.csv` |
| Procedure | `codebook_hier_w_embeddings_proc.csv` |
| Medication | `codebook_atc_hier_w_embeddings.csv` |

Each node codebook follows the schema `GlobalID,Layer,LayerTokenID,Code,String,0,...,767`. The code validates that the node rows align with the codebook reconstructed from the hierarchy.

```bash
NODE_CODEBOOK_DIR="/path/to/node_codebooks"
python smi.py reproduce --node-codebooks "$NODE_CODEBOOK_DIR"
```

## Outputs

By default, `reproduce` writes the CSVs under `outputs/`.

| Domain | Five-level ontology | SMI embeddings for complete codes |
| --- | --- | --- |
| Diagnosis | `diagnosis_hierarchy.csv` | `diagnosis_SMI_embeddings.csv` |
| Procedure | `procedure_hierarchy.csv` | `procedure_SMI_embeddings.csv` |
| Medication | `medication_hierarchy.csv` | `medication_SMI_embeddings.csv` |

The hierarchy CSVs record the codes and labels along each ontology path. The embedding CSVs contain `Code`, a dot-separated `SemanticID`, and columns `0` through `767`. The L1–L4 tables contain sums along the path up to their respective levels. They are stored in `outputs/smi_embeddings_by_level/` with names such as `diagnosis_SMI_embeddings_L1.csv`.

Retrieve the SMI embedding for ATC code `A01AA01`.

```python
import pandas as pd

table = pd.read_csv(
    "outputs/medication_SMI_embeddings.csv",
    dtype={"Code": str, "SemanticID": str},
)
row = table.loc[table["Code"].eq("A01AA01")].iloc[0]
semantic_id = row["SemanticID"]
embedding = row[[str(i) for i in range(768)]].to_numpy(dtype=float)
```

## Run individual stages

Build the hierarchies from the source mappings, then aggregate embeddings for a selected domain. The domain names on the command line are `diag`, `proc`, and `atc`.

```bash
python smi.py build-hierarchy
python smi.py aggregate diag \
  --node-codebook "$NODE_CODEBOOK_DIR/codebook_hier_w_embeddings_diag.csv"
```

The default procedure output preserves the historical Semantic ID convention, where the current node's token is empty at each layer. Use `--full-procedure-paths` to include the current node in the sum. The example below writes those outputs to a separate directory inside the repository.

```bash
python smi.py --output-dir full_procedure build-hierarchy --domain proc
python smi.py --output-dir full_procedure aggregate proc --full-procedure-paths \
  --node-codebook "$NODE_CODEBOOK_DIR/codebook_hier_w_embeddings_proc.csv"
```

For scripts that expect the legacy ICD codebook, run the following command after diagnosis aggregation.

```bash
python smi.py export-eval-codebook diag \
  --node-codebook "$NODE_CODEBOOK_DIR/codebook_hier_w_embeddings_diag.csv"
```

## Ontology sources

- **Diagnosis** uses the Phecode–ICD vocabulary, the AHRQ ICD-9 diagnosis mapping, the GraphCare ICD-10–CCS mapping, and the multilevel CCS labels and assignments in `inputs/diagnosis/`. [AHRQ ICD-9-CM CCS](https://hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp)
- **Procedure** uses AHRQ Appendices B and D for ICD-9, CCS v2020.1 for ICD-10-PCS, and the historical code descriptions, prefix labels, and CCS overrides in `inputs/procedure/`. [AHRQ ICD-9-CM CCS](https://hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp) · [AHRQ ICD-10-PCS CCS](https://hcup-us.ahrq.gov/toolssoftware/ccs10/ccs10.jsp)
- **Medication** uses the WHO ATC-DDD snapshot dated July 31, 2024, in `inputs/medication/`.

## Citation

```bibtex
@inproceedings{song2026smi,
  title     = {{SMI}: Ontology-Grounded Semantic {IDs} for Hierarchy-Aware Medical Concept Representation},
  author    = {Song, Ziyang and Shen, Lia and Li, Yixuan and Lu, Qincheng and
               Li, Weihao and Xu, Haoran and Zhu, He and Wang, Xinyu and Li, Yue},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
