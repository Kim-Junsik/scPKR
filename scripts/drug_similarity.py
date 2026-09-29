"""Molecular structure of the combosciplex drugs: SMILES and pairwise Tanimoto.

    python scripts/drug_similarity.py              # recompute from assets/drugs/
    python scripts/drug_similarity.py --refetch    # query PubChem again first

Structure only. No expression data and no score is read, so nothing here can leak
a test result into a design decision.

Outputs, all under assets/drugs/:
  pubchem_smiles.json          raw PubChem PUG REST answer per drug name
  drug_smiles.csv              drug, PubChem CID, parent SMILES (largest fragment)
  tanimoto_ecfp4_2048.csv      Morgan radius 2, 2048 bits - the common default
  tanimoto_morgan_r4_1024.csv  radius 4, 1024 bits - what scDFM's
                               get_molecular_fingerprints computes
  fingerprints_<setting>.npz   drugs (sorted names) and bits (uint8, drugs x bits),
                               checked to reproduce the Tanimoto matrix exactly

Computing needs RDKit, which is not in requirements.txt: pip install rdkit.
Reading the saved fingerprints (load_fingerprints) needs numpy only.

Measured 2026-09-17 (ECFP4): 136 drug pairs, median 0.11, p90 0.20. Only four
groups stand above that - Alvespimycin-Tanespimycin 0.85, SRT1720-SRT2104 0.50,
Dacinostat-Panobinostat 0.43, Givinostat-PCI-34051 0.35. Crizotinib, Curcumin and
Sorafenib have no analogue (max 0.14-0.23), and Panobinostat-Givinostat, the two
HDAC inhibitors every test combination hinges on, sit at 0.16.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "assets", "drugs")

DRUGS = ["Alvespimycin", "Carmofur", "Cediranib", "Crizotinib", "Curcumin", "Dacinostat",
         "Danusertib", "Dasatinib", "Givinostat", "PCI-34051", "Panobinostat", "Pirarubicin",
         "SRT1720", "SRT2104", "SRT3025", "Sorafenib", "Tanespimycin"]
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/{}/JSON"

# Roles, for the printed summary only.
TRAIN_SINGLE = {"Dasatinib", "Givinostat", "Panobinostat", "SRT2104"}
TEST_SINGLE = ["Dacinostat", "Alvespimycin"]
TEST_PARTNER = ["Crizotinib", "Curcumin", "SRT1720", "Sorafenib"]
VALIDATION_SINGLE = ["Dasatinib", "SRT2104"]

SETTINGS = {
    "ecfp4_2048": dict(radius=2, fpSize=2048),
    "morgan_r4_1024": dict(radius=4, fpSize=1024),
}


def fetch(name: str) -> dict:
    quoted = urllib.parse.quote(name)
    last = "no response"
    # PubChem renamed IsomericSMILES to SMILES in 2025; the new name is tried first.
    for props in ("SMILES,ConnectivitySMILES,MolecularFormula,Title",
                  "IsomericSMILES,CanonicalSMILES,MolecularFormula,Title"):
        try:
            with urllib.request.urlopen(PUBCHEM.format(quoted, props), timeout=30) as response:
                rows = json.load(response)["PropertyTable"]["Properties"]
            return {"query": name, "n_hits": len(rows), "hits": rows}
        except urllib.error.HTTPError as error:
            last = f"HTTP {error.code} for {props}"
    return {"query": name, "error": last}


def refetch() -> None:
    os.makedirs(OUT, exist_ok=True)
    results = []
    for name in DRUGS:
        result = fetch(name)
        results.append(result)
        status = result.get("error") or f"{result['n_hits']} hit(s), CID {result['hits'][0]['CID']}"
        print(f"  {name:14s} {status}")
        time.sleep(0.3)  # PubChem allows at most 5 requests per second
    with open(os.path.join(OUT, "pubchem_smiles.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def load_molecules() -> dict:
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError as error:
        raise SystemExit("RDKit is required: pip install rdkit") from error
    with open(os.path.join(OUT, "pubchem_smiles.json"), encoding="utf-8") as handle:
        records = json.load(handle)
    chooser = rdMolStandardize.LargestFragmentChooser()
    molecules = {}
    for record in records:
        if "error" in record or record["n_hits"] != 1:
            raise SystemExit(f"{record['query']}: expected exactly one PubChem hit, got {record}")
        hit = record["hits"][0]
        smiles = hit.get("SMILES") or hit.get("IsomericSMILES")
        parent = chooser.choose(Chem.MolFromSmiles(smiles))  # drops counter-ions such as HCl
        molecules[record["query"]] = {"cid": hit["CID"], "parent": Chem.MolToSmiles(parent),
                                      "mol": parent}
    return molecules


def similarity(molecules: dict, radius: int,
               fpSize: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Drug names, the Tanimoto matrix, and the fingerprint bits themselves."""
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
    names = sorted(molecules)
    fingerprints = [generator.GetFingerprint(molecules[n]["mol"]) for n in names]
    bits = np.array([list(fp) for fp in fingerprints], dtype=np.uint8)
    matrix = np.array([DataStructs.BulkTanimotoSimilarity(fp, fingerprints)
                       for fp in fingerprints])
    return names, matrix, bits


def load_fingerprints(label: str = "ecfp4_2048") -> tuple[list[str], np.ndarray]:
    """Drug names and fingerprint bits from assets/drugs, without RDKit.

    The model reads these; training hosts need numpy only.
    """
    with np.load(os.path.join(OUT, f"fingerprints_{label}.npz")) as archive:
        return [str(n) for n in archive["drugs"]], archive["bits"].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refetch", action="store_true", help="query PubChem again")
    args = parser.parse_args()

    if args.refetch or not os.path.exists(os.path.join(OUT, "pubchem_smiles.json")):
        print("querying PubChem")
        refetch()
    molecules = load_molecules()
    with open(os.path.join(OUT, "drug_smiles.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["drug", "pubchem_cid", "parent_smiles"])
        for name in sorted(molecules):
            writer.writerow([name, molecules[name]["cid"], molecules[name]["parent"]])

    for label, params in SETTINGS.items():
        names, matrix, bits = similarity(molecules, **params)
        np.savetxt(os.path.join(OUT, f"tanimoto_{label}.csv"), matrix, delimiter=",",
                   header=",".join(names), fmt="%.4f", comments="")
        np.savez_compressed(os.path.join(OUT, f"fingerprints_{label}.npz"),
                            drugs=np.array(names), bits=bits)
        # The saved bits must reproduce the RDKit similarity exactly.
        inter = bits.astype(np.int64) @ bits.T.astype(np.int64)
        counts = bits.sum(axis=1).astype(np.int64)
        union = counts[:, None] + counts[None, :] - inter
        if not np.allclose(inter / np.maximum(union, 1), matrix, atol=1e-9):
            raise SystemExit(f"{label}: saved fingerprint bits do not reproduce Tanimoto")
        index = {n: i for i, n in enumerate(names)}
        upper = matrix[np.triu_indices(len(names), k=1)]
        print(f"\n{label}: {len(upper)} pairs, median {np.median(upper):.3f}, "
              f"p90 {np.percentile(upper, 90):.3f}")
        for name in TEST_SINGLE + TEST_PARTNER + VALIDATION_SINGLE:
            row = matrix[index[name]].copy()
            row[index[name]] = -1.0
            nearest = names[int(np.argmax(row))]
            anchored = max((c for c in TRAIN_SINGLE if c != name),
                           key=lambda c: matrix[index[name], index[c]])
            print(f"  {name:13s} nearest {nearest} {row.max():.2f}   "
                  f"nearest with a training single {anchored} "
                  f"{matrix[index[name], index[anchored]]:.2f}")


if __name__ == "__main__":
    main()
