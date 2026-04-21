#!/usr/bin/env python3
"""
silva_assemble_db.py
Produces the same outputs as assemble_db.py but reads from a Kraken2/SILVA 138 database dir:
  - taxonomy_index.tsv
  - sequences_by_category/  (one FASTA per lowest-rank node)

Usage:
  python silva_assemble_db.py --db k2_16S/ --outdir kraken_db_converted/sequences_by_category/ --tax-out kraken_db_converted/taxonomy_index.tsv
"""

import argparse
from collections import defaultdict
from pathlib import Path

# SILVA rank names → your prefix convention
RANK_TO_PREFIX = {
    "domain":  "k__",
    "phylum":  "p__",
    "class":   "c__",
    "order":   "o__",
    "family":  "f__",
    "genus":   "g__",
    "species": "s__",
}

def is_valid_bacterial(tax_string: str) -> bool:
    return tax_string.startswith("Bacteria;")

SKIP_SPECIES = {"unidentified", "uncultured", "unclassified"}

def safe_name(label: str) -> str:
    return (
        label
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )

def load_rank_lookup(tax_txt: Path) -> dict:
    """
    Parses tax_slv_ssu_138.1.txt into a dict:
      bare_path (e.g. "Bacteria;Firmicutes;") → rank string (e.g. "phylum")
    """
    lookup = {}
    with open(tax_txt) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            bare_path = parts[0]   # e.g. "Bacteria;Firmicutes;"
            rank      = parts[2]   # e.g. "phylum"
            lookup[bare_path] = rank
    return lookup

def build_prefixed_taxonomy(tax_string: str, rank_lookup: dict) -> str:
    tokens = [t for t in tax_string.split(";") if t.strip()]

    try:
        start = tokens.index("Bacteria")
    except ValueError:
        return None  # caught by caller

    bacterial_tokens = tokens[start:]
    prefixed = ["k__Bacteria"]
    cumulative = "Bacteria;"

    for i, token in enumerate(bacterial_tokens[1:], 1):
        is_last = (i == len(bacterial_tokens) - 1)
        cumulative += token + ";"

        if is_last:
            if token.lower() in SKIP_SPECIES:
                pass  # just don't append — genus becomes the lowest rank
            else:
                prefixed.append(f"s__{token}")
        else:
            if token.lower() in SKIP_SPECIES:
                return None  # unidentified at an intermediate rank — discard whole record
            rank = rank_lookup.get(cumulative)
            prefix = RANK_TO_PREFIX.get(rank)
            if prefix:
                prefixed.append(f"{prefix}{token}")

    return ";".join(prefixed)

def parse_lowest_rank(prefixed_tax: str) -> str:
    fields = [f.strip() for f in prefixed_tax.split(";")]
    for f in reversed(fields):
        if "__" in f and not f.endswith("__"):
            return f
    return "unclassified"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to k2_16S/ directory")
    parser.add_argument("--tax-out", default="taxonomy_index.tsv")
    parser.add_argument("--outdir", default="sequences_by_category")
    args = parser.parse_args()

    db = Path(args.db)
    fasta_path  = db / "library" / "silva.fna"
    tax_txt     = db / "data"    / "tax_slv_ssu_138.1.txt"
    outdir      = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    print("Loading rank lookup...")
    rank_lookup = load_rank_lookup(tax_txt)

    # --- Parse FASTA, build taxonomy maps, bucket sequences ---
    id_to_tax = {}
    id_to_low = {}
    buffers   = defaultdict(list)

    print("Parsing sequences...")
    with open(fasta_path) as f:
        seq_id  = None
        tax_str = None
        seq     = []

        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                # Flush previous record
                if seq_id is not None:
                    buffers[id_to_low[seq_id]].append((seq_id, "".join(seq)))

                # Reset sentinel before deciding whether to keep this record
                seq_id = None
                seq = []

                header  = line[1:]
                parts   = header.split(" ", 1)
                raw_tax = parts[1] if len(parts) > 1 else ""

                if not is_valid_bacterial(raw_tax):
                    continue

                prefixed = build_prefixed_taxonomy(raw_tax, rank_lookup)
                if prefixed is None:
                    continue

                sid    = parts[0]
                lowest = parse_lowest_rank(prefixed)
                id_to_tax[sid] = prefixed
                id_to_low[sid] = lowest
                seq_id = sid  # only set if record passed all filters
            else:
                seq.append(line)

        # Flush last record
        if seq_id is not None:
            buffers[id_to_low[seq_id]].append((seq_id, "".join(seq)))

    # --- Create empty FASTAs for every node at every rank ---
    all_nodes = set()
    for prefixed_tax in id_to_tax.values():
        fields = [f.strip() for f in prefixed_tax.split(";")]
        for field in fields:
            if "__" in field and not field.endswith("__"):
                all_nodes.add(field)

    for node in all_nodes:
        (outdir / f"{safe_name(node)}.fasta").touch(exist_ok=True)

    # --- Write taxonomy index ---
    print(f"Writing {args.tax_out}...")
    with open(args.tax_out, "w") as out:
        for seq_id in id_to_tax:
            out.write(f"{seq_id}\t{id_to_low[seq_id]}\t{id_to_tax[seq_id]}\n")

    # --- Write bucketed FASTAs ---
    print(f"Writing sequences to {args.outdir}/...")
    for cat, records in buffers.items():
        fname = outdir / f"{safe_name(cat)}.fasta"
        with open(fname, "w") as out:
            for sid, s in records:
                out.write(f">{sid}\n{s}\n")

    print(f"Done. {len(id_to_tax)} sequences → {len(all_categories)} categories.")

if __name__ == "__main__":
    main()