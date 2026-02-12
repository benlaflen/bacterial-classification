#!/usr/bin/env python3

import argparse
import random
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict
import time
import sys

def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)
# -----------------------------
# helpers
# -----------------------------

def parse_fasta(path):
    seqs = {}
    with open(path) as f:
        sid = None
        buf = []
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if sid is not None:
                    seqs[sid] = "".join(buf)
                sid = line[1:].split()[0].strip()
                buf = []
            else:
                buf.append(line)
        if sid is not None:
            seqs[sid] = "".join(buf)
    return seqs

def parse_taxonomy_index(path):
    tax = {}
    with open(path) as f:
        f.readline()
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")

            if len(parts) < 3:
                raise RuntimeError("Taxonomy index must be tab-delimited with 3 columns")

            feature_id = parts[0].strip()
            taxonomy = parts[2].strip()

            tax[feature_id] = taxonomy

    return tax

def write_fasta(records, path):
    with open(path, "w") as out:
                for sid, seq in records:
                    out.write(f">{sid}\n{seq}\n")


def read_fastq(path):
    with open(path) as f:
        while True:
            h = f.readline()
            if not h:
                break
            s = f.readline()
            p = f.readline()
            q = f.readline()
            if not q:
                raise RuntimeError(f"Truncated FASTQ: {path}")
            yield h.rstrip(), s.rstrip(), q.rstrip()


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# -----------------------------
# main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    # inputs
    ap.add_argument("--seq-dir", default="sequences_by_category")
    ap.add_argument("--num-taxa", type=int, default=25)
    ap.add_argument("--total-reads", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)

    # ART (art_modern) params
    ap.add_argument("--art", default="./art_modern/opt/build_release/art_modern", help="Path to art_modern binary")
    ap.add_argument("--qual-profile", default="./art_modern/data/Illumina_profiles/HiSeq2500L125R1.txt", help="Illumina quality profile file")
    ap.add_argument("--read-len", type=int, default=125)
    ap.add_argument("--parallel", type=int, default=4)

    # trimming (post-ART length variability)
    ap.add_argument("--min-read-len", type=int, default=90)
    ap.add_argument("--max-read-len", type=int, default=125)

    # template control
    ap.add_argument("--max-templates-per-taxon", type=int, default=50)

    # outputs
    ap.add_argument("--out-prefix", default="sample")

    ap.add_argument("--taxonomy-index", default="./taxonomy_index.tsv",
                help="TSV mapping template ID to full taxonomy path")

    args = ap.parse_args()
    random.seed(args.seed)

    taxonomy_lookup = {}
    if args.taxonomy_index:
        log(f"Loading taxonomy index: {args.taxonomy_index}")
        taxonomy_lookup = parse_taxonomy_index(args.taxonomy_index)
    log(f"Loaded {len(taxonomy_lookup)} taxonomy entries")

    log("Starting mixed-sample generation")
    log(f"Seed = {args.seed}")
    log(f"Num taxa = {args.num_taxa}")
    log(f"Total reads = {args.total_reads}")

    seq_dir = Path(args.seq_dir)
    categories = sorted(p for p in seq_dir.iterdir() if p.suffix == ".fasta" and p.name.startswith("s__"))

    if len(categories) < args.num_taxa:
        raise SystemExit("Not enough categories available")

    present = random.sample(categories, args.num_taxa)

    # Dirichlet(1) abundance
    weights = [random.gammavariate(1.0, 1.0) for _ in range(args.num_taxa)]
    total = sum(weights)
    fracs = [w / total for w in weights]

    reads_per_taxon = [max(1, int(f * args.total_reads)) for f in fracs]
    while sum(reads_per_taxon) < args.total_reads:
        reads_per_taxon[random.randrange(len(reads_per_taxon))] += 1

    out_fastq = open(f"{args.out_prefix}.fastq", "w")
    out_fasta = open(f"{args.out_prefix}.fasta", "w")
    out_meta = open(f"{args.out_prefix}_metadata.tsv", "w")
    out_meta.write("read_id\ttaxonomy\ttemplate_id\torig_len\tfinal_len\n")

    read_counter = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for i, (cat_path, n_reads) in enumerate(zip(present, reads_per_taxon), 1):
            log(f"[{i}/{args.num_taxa}] Category {cat_path.stem}")
            log(f"  Target reads = {n_reads}")
            category = cat_path.stem
            seqs = parse_fasta(cat_path)

            template_ids = list(seqs.keys())
            if len(template_ids) > args.max_templates_per_taxon:
                template_ids = random.sample(template_ids, args.max_templates_per_taxon)

            # Rename template IDs to avoid '-' parsing issues
            tpl_to_orig = {}
            tpl_to_tax = {}
            templates = []
            for j, orig_id in enumerate(template_ids, 1):
                tpl_id = f"tpl{j:06d}"
                tpl_to_orig[tpl_id] = orig_id
                tpl_to_tax[tpl_id] = taxonomy_lookup.get(orig_id, "NA")
                if tpl_to_tax == "NA": print("id: '" + orig_id + "'")
                templates.append((tpl_id, seqs[orig_id]))

            tmp_fa = tmpdir / "templates.fa"
            write_fasta(templates, tmp_fa)

            total_template_len = sum(len(seq) for _, seq in templates)
            fcov = (n_reads * args.read_len) / total_template_len
            log(f"  Templates used = {len(templates)}")
            log(f"  fcov = {fcov:.4f}")
            out_prefix = tmpdir / "art_out"

            cmd = [
                args.art,
                "--mode", "wgs",
                "--lc", "se",
                "--qual_file_1", args.qual_profile,
                "--read_len", str(args.read_len),
                "--parallel", str(args.parallel),
                "--i-fcov", f"{fcov:.4f}",
                "--i-file", str(tmp_fa),
                "--o-fastq", str(out_prefix),
            ]

            run(cmd)

            fq_path = Path(f"{out_prefix}")
            if not fq_path.exists():
                raise RuntimeError("ART output FASTQ not found")

            for h, seq, qual in read_fastq(fq_path):
                raw = h[1:].split()[0]
                tpl_id = raw.split(":")[0]           # safe now
                orig_template = tpl_to_orig.get(tpl_id, tpl_id)
                taxonomy = tpl_to_tax.get(tpl_id, "NA")
                if taxonomy == "NA": print("tax: '" + tpl_id + "'")

                final_len = random.randint(args.min_read_len, args.max_read_len)
                if final_len > len(seq):
                    continue

                seq = seq[:final_len]
                qual = qual[:final_len]

                read_counter += 1
                rid = f"read_{read_counter:08d}"

                out_fastq.write(f"@{rid}\n{seq}\n+\n{qual}\n")
                out_fasta.write(f">{rid}\n{seq}\n")
                out_meta.write(
                    f"{rid}\t{taxonomy}\t{orig_template}\t{args.read_len}\t{final_len}\n"
                )

    out_fastq.close()
    out_meta.close()
    out_fasta.close()

    print(f"Generated {read_counter} reads from {args.num_taxa} taxa.")
    print(f"Outputs:")
    print(f"  {args.out_prefix}.fastq")
    print(f"  {args.out_prefix}_metadata.tsv")
    print(f"  {args.out_prefix}.fasta")


if __name__ == "__main__":
    main()