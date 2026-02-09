#!/usr/bin/env python3

from collections import defaultdict
from pathlib import Path

FASTA = "dna_sequences.fasta"
TAX = "taxonomy.tsv"
TAX_OUT = "taxonomy_index.tsv"
OUTDIR = Path("sequences_by_category")
OUTDIR.mkdir(exist_ok=True)

# -----------------------------
# helpers
# -----------------------------
def parse_lowest_rank(tax):
    # taxonomy fields are semicolon-separated
    # lowest rank = last non-empty, non-__ entry
    fields = [f.strip() for f in tax.split(";")]
    for f in reversed(fields):
        if "__" in f and not f.endswith("__"):
            return f
    return "unclassified"

# -----------------------------
# load taxonomy
# -----------------------------
id_to_tax = {}
id_to_low = {}

with open(TAX) as f:
    for line in f:
        seq_id, tax = line.rstrip().split("\t", 1)
        low = parse_lowest_rank(tax)
        id_to_tax[seq_id] = tax
        id_to_low[seq_id] = low

def safe_name(label: str) -> str:
    # ART (and many tools) break on spaces and special chars
    return (
        label
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )

# -----------------------------
# collect all taxonomy categories
# -----------------------------
all_categories = set()

def extract_all_ranks(tax):
    fields = [f.strip() for f in tax.split(";")]
    return [f for f in fields if "__" in f and not f.endswith("__")]

for tax in id_to_tax.values():
    for rank in extract_all_ranks(tax):
        all_categories.add(rank)

# -----------------------------
# create empty FASTA files for all categories
# -----------------------------
for category in all_categories:
    safe_category = safe_name(category)
    fname = OUTDIR / f"{safe_category}.fasta"
    fname.touch(exist_ok=True)

# -----------------------------
# write taxonomy index
# -----------------------------
with open(TAX_OUT, "w") as out:
    for seq_id in id_to_tax:
        out.write(f"{seq_id}\t{id_to_low[seq_id]}\t{id_to_tax[seq_id]}\n")

# -----------------------------
# bucket sequences
# -----------------------------
buffers = defaultdict(list)

with open(FASTA) as f:
    seq_id = None
    seq = []

    for line in f:
        line = line.rstrip()
        if line.startswith(">"):
            if seq_id is not None:
                buffers[id_to_low[seq_id]].append((seq_id, "".join(seq)))
            seq_id = line[1:].split()[0]
            seq = []
        else:
            seq.append(line)

    # last record
    if seq_id is not None:
        buffers[id_to_low[seq_id]].append((seq_id, "".join(seq)))

# -----------------------------
# write FASTA buckets
# -----------------------------
for category, records in buffers.items():
    safe_category = safe_name(category)
    fname = OUTDIR / f"{safe_category}.fasta"
    with open(fname, "w") as out:
        for sid, s in records:
            out.write(f">{sid}\n{s}\n")

