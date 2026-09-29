"""Download the KEGG snapshot the pathway mask is built from.

    python scripts/download_kegg.py
    python scripts/download_kegg.py --force

Three files from KEGG's REST API (free for academic use, no login):

    list/pathway/hsa   human pathway ids and names
    link/hsa/pathway   pathway <-> gene links
    list/hsa           gene entry -> symbol, for mapping onto adata.var_names

KEGG has no version endpoint, so the retrieval date is written next to the files.
Re-running later can therefore give a different snapshot; `manifest.json` records
line counts and byte sizes so a change is at least visible rather than silent.

For a fully pinned alternative, MSigDB ships the same content as a release-numbered
GMT in symbol form (c2.cp.kegg_medicus.v*.symbols.gmt), which also removes the
Entrez-to-symbol mapping step. It needs a login, so it cannot be fetched here.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module

ENDPOINTS = {
    "pathway_list.tsv": "https://rest.kegg.jp/list/pathway/hsa",
    "pathway_gene.tsv": "https://rest.kegg.jp/link/hsa/pathway",
    "gene_list.tsv": "https://rest.kegg.jp/list/hsa",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    config = config_module.load(args.overrides)
    target = config["data"]["kegg_dir"]
    os.makedirs(target, exist_ok=True)

    manifest = {"retrieved": datetime.date.today().isoformat(), "files": {}}
    for name, url in ENDPOINTS.items():
        path = os.path.join(target, name)
        if os.path.exists(path) and os.path.getsize(path) > 0 and not args.force:
            print(f"  {name:20s} exists ({os.path.getsize(path):,} B) - use --force to refresh")
        else:
            print(f"  {name:20s} downloading from {url} ...")
            urllib.request.urlretrieve(url, path)
        with open(path, encoding="utf-8") as handle:
            lines = sum(1 for _ in handle)
        manifest["files"][name] = {"bytes": os.path.getsize(path), "lines": lines}
        print(f"  {'':20s} {lines:,} lines, {os.path.getsize(path):,} B")

    with open(os.path.join(target, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\n-> {target}")
    print("   reference snapshot (2026-08-12): pathway_list 372, pathway_gene 39,574, "
          "gene_list 24,252 lines")
    print("   a different count means KEGG changed; the mask K and its density will move")


if __name__ == "__main__":
    main()
