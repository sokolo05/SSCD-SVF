#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pysam
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import os
import math
import random
from typing import Optional

# ------------------------------
# PositionMapping / ReadReconstruction
# ------------------------------
op_to_char = {
    0: 'M', 1: 'I', 2: 'D', 3: 'N', 4: 'S', 5: 'H', 6: 'P', 7: '=', 8: 'X'
}

base_to_num = {'A':1,'C':2,'G':3,'T':4,'N':0,'-':0,'*':0,'?':0}

class PositionMapping:
    def __init__(self):
        self.mappings = []
    def add_mapping(self, recon_pos: int, ref_pos: Optional[int], read_pos: Optional[int], op_type: str, base: str):
        self.mappings.append({'recon_pos': recon_pos, 'ref_pos': ref_pos, 'read_pos': read_pos, 'op_type': op_type, 'base': base})
    def find_by_ref_pos(self, ref_pos: int):
        for mapping in self.mappings:
            if mapping['ref_pos'] == ref_pos:
                return mapping
        return None

class GlobalReadInfo:
    def __init__(self, read):
        self.read = read
        self.cigartuples = read.cigartuples
        self.reference_start = read.reference_start
        self.query_sequence = read.query_sequence
        self.reference_end = read.reference_end
        self.mapping_quality = read.mapping_quality
        self.reference_name = read.reference_name

class ReadReconstruction:
    def __init__(self, global_read: GlobalReadInfo, ref: pysam.FastaFile, chrom: str, target_pos: int):
        self.global_info = global_read
        self.read = global_read.read
        self.chrom = chrom
        self.target_pos = target_pos
        self.target_pos_0based = target_pos - 1
        self.ref = ref
        self.reconstructed_bases = []
        self.reconstructed_cigar = []
        self.position_mapping = PositionMapping()
        self.reconstruction_length = 0
        self._reconstruct_both_sequences()

    def _reconstruct_both_sequences(self):
        read = self.read
        cigartuples = read.cigartuples or []
        query_seq = read.query_sequence or ''
        query_pos = 0
        ref_pos = read.reference_start
        recon_pos = 0

        for op, op_len in cigartuples:
            op_char = op_to_char.get(op, '?')
            if op in (0,7,8):  # M,=,X
                for i in range(op_len):
                    base = query_seq[query_pos + i] if (query_pos + i) < len(query_seq) else 'N'
                    self.reconstructed_bases.append(base)
                    self.reconstructed_cigar.append(op_char)
                    self.position_mapping.add_mapping(recon_pos, ref_pos + i, query_pos + i, op_char, base)
                    recon_pos += 1
                query_pos += op_len
                ref_pos += op_len
            elif op == 1:  # I
                for i in range(op_len):
                    base = query_seq[query_pos + i] if (query_pos + i) < len(query_seq) else 'N'
                    self.reconstructed_bases.append(base)
                    self.reconstructed_cigar.append('I')
                    self.position_mapping.add_mapping(recon_pos, None, query_pos + i, 'I', base)
                    recon_pos += 1
                query_pos += op_len
            elif op in (2,3):  # D or N
                for i in range(op_len):
                    try:
                        ref_base = self.ref.fetch(self.chrom, ref_pos + i, ref_pos + i + 1).upper()
                    except:
                        ref_base = 'N'
                    self.reconstructed_bases.append(ref_base)
                    self.reconstructed_cigar.append('D' if op == 2 else 'N')
                    self.position_mapping.add_mapping(recon_pos, ref_pos + i, None, 'D' if op == 2 else 'N', ref_base)
                    recon_pos += 1
                ref_pos += op_len
            elif op == 4:  # S
                for i in range(op_len):
                    base = query_seq[query_pos + i] if (query_pos + i) < len(query_seq) else 'N'
                    self.reconstructed_bases.append(base)
                    self.reconstructed_cigar.append('S')
                    self.position_mapping.add_mapping(recon_pos, None, query_pos + i, 'S', base)
                    recon_pos += 1
                query_pos += op_len
            elif op == 5:  # H
                continue
            elif op == 6:  # P
                for i in range(op_len):
                    self.reconstructed_bases.append('*')
                    self.reconstructed_cigar.append('P')
                    self.position_mapping.add_mapping(recon_pos, None, None, 'P', '*')
                    recon_pos += 1

        self.reconstruction_length = recon_pos
        if len(self.reconstructed_bases) != len(self.reconstructed_cigar):
            raise ValueError(f"Reconstruction length mismatch: bases {len(self.reconstructed_bases)} vs cigar {len(self.reconstructed_cigar)}")

    def get_target_subsequences(self, extend_length: int):
        mapping = self.position_mapping.find_by_ref_pos(self.target_pos_0based)
        if mapping is None:
            return None, None
        target_recon_pos = mapping['recon_pos']
        left = max(0, target_recon_pos - extend_length)
        right = min(self.reconstruction_length, target_recon_pos + extend_length + 1)
        bases_subseq = self.reconstructed_bases[left:right]
        cigar_subseq = self.reconstructed_cigar[left:right]
        pad_left = max(0, extend_length - (target_recon_pos - left))
        pad_right = max(0, extend_length - (right - target_recon_pos - 1))
        bases_subseq = ['-'] * pad_left + bases_subseq + ['-'] * pad_right
        cigar_subseq = ['-'] * pad_left + cigar_subseq + ['-'] * pad_right
        if len(bases_subseq) != len(cigar_subseq):
            raise ValueError("Subsequence length mismatch")
        return bases_subseq, cigar_subseq

# ------------------------------
# === Color Mapping ===
# ------------------------------
cmap_cigar = {
    2: [1.0, 0.0, 0.0],     # I
    4: [1.0, 0.0, 1.0],     # S
    3: [0.0, 1.0, 0.0],     # D/N
    1: [0.5, 0.5, 0.5],     # M/=/X
    0: [0.85, 0.85, 0.85]   # gap/other
}
cmap_base = {
    1: [1,0,0],    # A
    4: [1,0.8,0],  # T
    2: [0,1,0],    # C
    3: [0,0,1],    # G
    0: [0.5,0.5,0.5]
}
cmap_base_gc_merged = {
    1: [1,0,0],
    4: [1,0.8,0],
    2: [0.3,0.7,0.3],
    3: [0.3,0.7,0.3],
    0: [0.5,0.5,0.5]
}

# ------------------------------
# === Utility Functions: Statistics (GC, continuity, entropy) ===
# ------------------------------
def calc_gc_ratio_from_numeric_list(num_list):
    # num_list: list of ints 1..4 (0 means gap/other), skip zeros
    total = sum(1 for x in num_list if x in [1,2,3,4])
    if total == 0:
        return 0.0
    gc = sum(1 for x in num_list if x in [2,3])
    return gc / total

def calc_continuity_ratio_from_numeric_list(num_list):
    # continuity_count counts how many positions equal to previous (only ATCG)
    continuity_count = 0
    total_positions = 0
    prev = None
    for v in num_list:
        if v in [1,2,3,4]:
            if prev is not None and v == prev:
                continuity_count += 1
            prev = v
            total_positions += 1
        else:
            prev = None
    if total_positions == 0:
        return 0.0
    return continuity_count / total_positions

def calculate_entropy_numeric(num_list):
    counts = {'A':0,'C':0,'G':0,'T':0}
    total = 0
    for v in num_list:
        if v == 1:
            counts['A'] += 1; total += 1
        elif v == 2:
            counts['C'] += 1; total += 1
        elif v == 3:
            counts['G'] += 1; total += 1
        elif v == 4:
            counts['T'] += 1; total += 1
    if total == 0:
        return 0.0
    ent = 0.0
    for b in ['A','C','G','T']:
        p = counts[b] / total
        if p > 0:
            ent -= p * math.log2(p)
    return ent

# ------------------------------
# === Image Generation Functions (using numpy rgb arrays) ===
# ------------------------------
def base_rgb_from_numeric_matrix(arr):
    rgb = np.zeros((*arr.shape,3), dtype=np.float32)
    for num, col in cmap_base.items():
        rgb[arr == num] = col
    return rgb

def cigar_rgb_from_cigar_matrix(cigar_mat):
    # cigar_mat: list of list of chars
    h = len(cigar_mat)
    w = len(cigar_mat[0]) if h>0 else 0
    rgb = np.zeros((h,w,3), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            c = cigar_mat[i][j]
            if c == 'I':
                rgb[i,j] = cmap_cigar[2]
            elif c == 'S':
                rgb[i,j] = cmap_cigar[4]
            elif c in ['D','N']:
                rgb[i,j] = cmap_cigar[3]
            elif c in ['M','=','X']:
                rgb[i,j] = cmap_cigar[1]
            else:
                rgb[i,j] = cmap_cigar[0]
    return rgb

def weighted_dual_rgb(base_rgb, cigar_rgb):
    return (base_rgb + cigar_rgb) / 2.0

def color_with_continuity(base_data_numeric):
    h = len(base_data_numeric)
    w = len(base_data_numeric[0]) if h>0 else 0
    rgb = np.zeros((h,w,3), dtype=np.float32)
    for i in range(h):
        current = None
        continuity_count = 0
        for j in range(w):
            val = base_data_numeric[i][j]
            if val == current and val != 0:
                continuity_count += 1
            else:
                current = val
                continuity_count = 0
            # base color
            if val == 1:
                base_color = np.array([1,0,0])
            elif val == 2:
                base_color = np.array([0,1,0])
            elif val == 3:
                base_color = np.array([0,0,1])
            elif val == 4:
                base_color = np.array([1,0.8,0])
            else:
                base_color = np.array([0.5,0.5,0.5])
                continuity_count = 0
            if continuity_count > 0:
                fade_factor = max(0.5, 1.0 - continuity_count * 0.1)
                rgb[i,j] = base_color * fade_factor
            else:
                rgb[i,j] = base_color
    return rgb

def color_by_entropy(base_data_numeric):
    h = len(base_data_numeric)
    w = len(base_data_numeric[0]) if h>0 else 0
    rgb = np.zeros((h,w,3), dtype=np.float32)
    entropies = []
    for i in range(h):
        ent = calculate_entropy_numeric(base_data_numeric[i])
        entropies.append(ent)
    if entropies:
        mx = max(entropies); mn = min(entropies)
        rng = mx - mn if mx > mn else 1.0
        for i in range(h):
            normalized = (entropies[i] - mn) / rng
            brightness = 1.0 - normalized * 0.7
            for j in range(w):
                v = base_data_numeric[i][j]
                if v == 1:
                    base_color = np.array([1,0,0])
                elif v == 2:
                    base_color = np.array([0,1,0])
                elif v == 3:
                    base_color = np.array([0,0,1])
                elif v == 4:
                    base_color = np.array([1,0.8,0])
                else:
                    base_color = np.array([0.5,0.5,0.5])
                rgb[i,j] = base_color * brightness
    return rgb, entropies

def color_with_gc_merged_and_statistics(base_data_numeric):
    h = len(base_data_numeric)
    w = len(base_data_numeric[0]) if h>0 else 0
    rgb = np.zeros((h,w,3), dtype=np.float32)
    gc_contents = []
    continuity_ratios = []
    entropies = []
    for i in range(h):
        row = base_data_numeric[i]
        gc_count = 0
        total_bases = 0
        continuity_count = 0
        total_positions = 0
        current_base = None
        entropy_counts = {'A':0,'C':0,'G':0,'T':0}
        entropy_total = 0
        for v in row:
            if v in [2,3]:
                gc_count += 1
            if v in [1,2,3,4]:
                total_bases += 1
            if v in [1,2,3,4]:
                if v == current_base:
                    continuity_count += 1
                current_base = v
                total_positions += 1
            if v == 1:
                entropy_counts['A'] += 1; entropy_total += 1
            elif v == 2:
                entropy_counts['C'] += 1; entropy_total += 1
            elif v == 3:
                entropy_counts['G'] += 1; entropy_total += 1
            elif v == 4:
                entropy_counts['T'] += 1; entropy_total += 1
            if v == 1:
                base_color = np.array([1,0,0])
            elif v == 4:
                base_color = np.array([1,0.8,0])
            elif v in (2,3):
                base_color = np.array([0.3,0.7,0.3])
            else:
                base_color = np.array([0.5,0.5,0.5])
            rgb[i, np.where(np.ones(w))[0].tolist() if False else slice(None)] = rgb[i, :]  # no-op to keep shape
            rgb[i, np.arange(w)] = rgb[i, np.arange(w)]  # no-op
            # assign per j below after loop (we'll reassign properly)
        # Now assign per element (we need to reassign properly)
        for j in range(w):
            v = row[j]
            if v == 1:
                rgb[i,j] = np.array([1,0,0])
            elif v == 4:
                rgb[i,j] = np.array([1,0.8,0])
            elif v in (2,3):
                rgb[i,j] = np.array([0.3,0.7,0.3])
            else:
                rgb[i,j] = np.array([0.5,0.5,0.5])
        gc_contents.append(gc_count / total_bases if total_bases>0 else 0.0)
        continuity_ratios.append(continuity_count / total_positions if total_positions>0 else 0.0)
        if entropy_total > 0:
            ent = 0.0
            for k in ['A','C','G','T']:
                p = entropy_counts[k] / entropy_total
                if p > 0:
                    ent -= p * math.log2(p)
        else:
            ent = 0.0
        entropies.append(ent)
    avg_gc = np.mean(gc_contents) if gc_contents else 0.0
    avg_cont = np.mean(continuity_ratios) if continuity_ratios else 0.0
    avg_ent = np.mean(entropies) if entropies else 0.0
    return rgb, avg_gc, avg_cont, avg_ent

# ------------------------------
# === Process region and generate data matrices ===
# ------------------------------
def gather_reads_for_region(bam_path, ref_path, chrom, pos, extend_length, select_read):
    samfile = pysam.AlignmentFile(bam_path, "rb")
    ref = pysam.FastaFile(ref_path)
    start = max(0, pos - extend_length)
    end = pos + extend_length + 1

    # collect candidate reconstructions with mapping quality
    recs = []
    for read in samfile.fetch(chrom, start, end):
        if read.is_unmapped:
            continue
        # only process reads that span the position in terms of reconstructed coords: we'll construct then check
        gri = GlobalReadInfo(read)
        try:
            rr = ReadReconstruction(gri, ref, chrom, pos)
        except Exception as e:
            # skip problematic reads
            continue
        bases_subseq, cigar_subseq = rr.get_target_subsequences(extend_length)
        if bases_subseq is None:
            continue
        # convert bases_subseq to numeric
        bases_num = []
        for b in bases_subseq:
            bu = b.upper() if isinstance(b, str) else '-'
            if bu == 'A': bases_num.append(1)
            elif bu == 'C': bases_num.append(2)
            elif bu == 'G': bases_num.append(3)
            elif bu == 'T': bases_num.append(4)
            else: bases_num.append(0)
        recs.append({
            'bases_num': bases_num,
            'cigar_seq': cigar_subseq,
            'mapq': getattr(read, 'mapping_quality', 0)
        })

    samfile.close()
    ref.close()

    # If no reads, return empty list
    if not recs:
        return [], []

    # Randomize tie-breaking, then sort by mapq desc
    random.shuffle(recs)
    recs.sort(key=lambda x: x['mapq'], reverse=True)

    # select top select_read
    if select_read is None or select_read <= 0:
        selected = recs
    else:
        selected = recs[:select_read]

    # If fewer than select_read, we will pad later in main (we return selected and total required)
    return selected, len(recs)

# ------------------------------
# === Main logic: generate images and save, generate csv ===
# ------------------------------
def save_rgb_image(rgb, out_path, dpi=300):
    plt.figure(frameon=False)
    ax = plt.Axes(plt.gcf(), [0.,0.,1.,1.])
    ax.set_axis_off()
    plt.gcf().add_axes(ax)
    ax.imshow(rgb, aspect='auto', interpolation='nearest')
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

def run_pipeline(txt_file, bam_file, ref_file, output_dir, extend_length, select_read, csv_path):
    os.makedirs(output_dir, exist_ok=True)
    # create subdirs for each image type
    subdirs = {}
    types = ['cigar','bases','dual','gc','entropy','gradual']
    for t in types:
        d = os.path.join(output_dir, t)
        os.makedirs(d, exist_ok=True)
        subdirs[t] = d

    # read input file (txt or csv)
    if txt_file.endswith('.txt'):
        df = pd.read_csv(txt_file, sep='\t', header=0, dtype=str)
    else:
        df = pd.read_csv(txt_file, header=0, dtype=str)

    # summary rows
    summary_rows = []

    for idx, row in df.iterrows():
        chrom = str(row['chr'])
        pos = int(row['pos'])
        svtype = str(row.get('sv_type','NA'))
        svlen = str(row.get('sv_len','NA'))
        identifier = f"{chrom}.{pos}.{svtype}.{svlen}"
        print(f"[INFO] Processing {identifier}")

        selected_recs, total_candidates = gather_reads_for_region(bam_file, ref_file, chrom, pos, extend_length, select_read)
        # determine seq_len from selected (they should have equal length)
        if selected_recs:
            seq_len = len(selected_recs[0]['bases_num'])
        else:
            # default seq length = 2*extend_length+1
            seq_len = 2 * extend_length + 1

        # create fixed number of reads = select_read (or total_candidates if select_read None)
        target_reads = select_read if (select_read and select_read>0) else max(len(selected_recs), 1)
        # If selected_recs length < target_reads, pad with N-only reads (0 numeric)
        padded_flags = []
        final_bases = []
        final_cigars = []

        # Use best selected first
        for rec in selected_recs:
            final_bases.append(rec['bases_num'])
            final_cigars.append(rec['cigar_seq'])
            padded_flags.append(False)
            if len(final_bases) >= target_reads:
                break

        # pad if necessary
        while len(final_bases) < target_reads:
            final_bases.append([0] * seq_len)  # N/gap representation numeric 0
            final_cigars.append(['-'] * seq_len)
            padded_flags.append(True)

        # Ensure all rows same length
        max_len = seq_len
        for i in range(len(final_bases)):
            if len(final_bases[i]) < max_len:
                final_bases[i].extend([0] * (max_len - len(final_bases[i])))
            if len(final_cigars[i]) < max_len:
                final_cigars[i].extend(['-'] * (max_len - len(final_cigars[i])))

        # Convert to numpy arrays where needed
        arr_bases = np.array(final_bases, dtype=int)
        # produce rgb for bases
        rgb_bases = base_rgb_from_numeric_matrix(arr_bases)
        # produce rgb for cigar
        rgb_cigar = cigar_rgb_from_cigar_matrix(final_cigars)
        # produce dual as average of base and cigar (per requirement, GC not involved)
        rgb_dual = weighted_dual_rgb(rgb_bases, rgb_cigar)
        # GC merged image and stats
        rgb_gc, avg_gc_ind, avg_cont_ind, avg_ent_ind = color_with_gc_merged_and_statistics(final_bases)
        # entropy image (and per-read entropies)
        rgb_entropy, entropies = color_by_entropy(final_bases)
        # gradual continuity image
        rgb_gradual = color_with_continuity(final_bases)

        # Save images to their subfolders with required naming scheme
        # File base name:
        base_fname = f"{identifier}"
        paths = {}
        for t, rgb_img in [('cigar', rgb_cigar), ('bases', rgb_bases), ('dual', rgb_dual),
                           ('gc', rgb_gc), ('entropy', rgb_entropy), ('gradual', rgb_gradual)]:
            out_path = os.path.join(subdirs[t], f"{base_fname}.{t}.png")
            save_rgb_image(rgb_img, out_path)
            paths[t] = out_path

        # Compute statistics (only from non-padded reads)
        valid_bases = []
        for i in range(len(final_bases)):
            if not padded_flags[i]:
                valid_bases.append(final_bases[i])
        # If no valid reads, stats are zeros
        if valid_bases:
            gc_list = [calc_gc_ratio_from_numeric_list(r) for r in valid_bases]
            cont_list = [calc_continuity_ratio_from_numeric_list(r) for r in valid_bases]
            ent_list = [calculate_entropy_numeric(r) for r in valid_bases]
            avg_gc = float(np.mean(gc_list))
            avg_cont = float(np.mean(cont_list))
            avg_ent = float(np.mean(ent_list))
        else:
            avg_gc = 0.0; avg_cont = 0.0; avg_ent = 0.0

        # Round to 4 decimal places for CSV
        avg_gc_f = f"{avg_gc:.4f}"
        avg_cont_f = f"{avg_cont:.4f}"
        avg_ent_f = f"{avg_ent:.4f}"

        # Append summary row
        summary_rows.append({
            'id': identifier,
            'avg_gc': avg_gc_f,
            'avg_continuity': avg_cont_f,
            'avg_entropy': avg_ent_f,
            'num_reads_used': len(valid_bases),
            'total_candidates': total_candidates,
            'path_bases': paths['bases'],
            'path_cigar': paths['cigar'],
            'path_dual': paths['dual'],
            'path_gc': paths['gc'],
            'path_entropy': paths['entropy'],
            'path_gradual': paths['gradual']
        })

        print(f"  Saved images for {identifier} (reads used for stats: {len(valid_bases)})")

    # write CSV
    summary_df = pd.DataFrame(summary_rows, columns=[
        'id','avg_gc','avg_continuity','avg_entropy','num_reads_used','total_candidates',
        'path_bases','path_cigar','path_dual','path_gc','path_entropy','path_gradual'
    ])
    summary_df.to_csv(csv_path, index=False)
    print(f"[INFO] Summary CSV written to: {csv_path}")
    print("[INFO] All done.")

# ------------------------------
# === CLI ===
# ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate SV visualizations (6 types) using ReadReconstruction.")
    p.add_argument('--txt_file', required=True, help="Input txt/csv with columns chr,pos,sv_type,sv_len")
    p.add_argument('--bam_file', required=True, help="Input BAM")
    p.add_argument('--ref_file', required=True, help="Reference FASTA")
    p.add_argument('--output_dir', required=True, help="Output directory (subdirs for each image type will be created)")
    p.add_argument('--extend_length', type=int, default=500, help="Extend length (default 500)")
    p.add_argument('--select_read', type=int, default=30, help="Max reads to select per site (pad with N if fewer).")
    p.add_argument('--csv_out', default='summary_stats.csv', help="CSV output path (inside output_dir by default)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    csv_path = args.csv_out
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(args.output_dir, csv_path)
    run_pipeline(args.txt_file, args.bam_file, args.ref_file, args.output_dir, args.extend_length, args.select_read, csv_path)
