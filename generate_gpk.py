#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integrated Version: ReadReconstruction analysis based on Code B, generating statistical CSV files
No image generation, only CSV statistical reports - version with 2 decimal places for floating point numbers
"""
import pysam
import argparse
import numpy as np
import pandas as pd
import os
import math
from typing import Optional, List, Dict, Tuple
from collections import Counter, defaultdict

# ------------------------------
# === Utility Functions: Numerical Formatting ===
# ------------------------------
def format_float(value, decimals=2):
    """Format floating-point number with specified decimal places"""
    if value is None:
        return 0.0
    try:
        # Convert to float first, then format
        float_value = float(value)
        # Use string formatting to avoid floating-point precision issues
        return float(f"{float_value:.{decimals}f}")
    except (ValueError, TypeError):
        return 0.0

# ------------------------------
# === Utility Functions: Repeat Sequence Analysis ===
# ------------------------------
def calculate_repeat_content(sequence: str, k: int = 3, freq_threshold: int = 2):
    """
    Calculate repeat sequence content of a sequence
    Returns: (repeat length, repeat ratio)
    """
    if not sequence or len(sequence) < k:
        return 0, 0.0
    
    total_len = len(sequence)
    kmer_counts = {}
    
    # Count k-mer frequencies
    for i in range(total_len - k + 1):
        kmer = sequence[i:i+k]
        kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
    
    # Identify high-frequency k-mers (potential repeats)
    repeat_regions = 0
    for i in range(total_len - k + 1):
        if kmer_counts.get(sequence[i:i+k], 0) >= freq_threshold:
            repeat_regions += 1
    
    repeat_length = repeat_regions + k - 1 if repeat_regions > 0 else 0
    repeat_ratio = repeat_length / total_len if total_len > 0 else 0.0
    
    return repeat_length, format_float(repeat_ratio, 2)

# ------------------------------
# === Utility Functions: Sequence Complexity ===
# ------------------------------
def calculate_complexity(sequence: str) -> float:
    """Calculate sequence complexity"""
    if not sequence or len(sequence) < 3:
        return 0.0
    # Filter out 'N' and '-'
    valid_sequence = [base for base in sequence if base.upper() not in ['N', '-', '*', '?']]
    if len(valid_sequence) < 3:
        return 0.0
    valid_seq_str = ''.join(valid_sequence)
    kmer_counts = Counter([valid_seq_str[i:i+3] for i in range(len(valid_seq_str)-2)])
    complexity = len(kmer_counts) / len(valid_seq_str)
    return format_float(complexity, 2)

def calculate_sequence_entropy(sequence: str) -> float:
    """Calculate sequence entropy"""
    if not sequence:
        return 0.0
    # Only consider ATCG bases, filter out 'N', '-', etc.
    valid_bases = [base for base in sequence.upper() if base in 'ATCG']
    if not valid_bases:
        return 0.0
    base_counts = Counter(valid_bases)
    total = len(valid_bases)
    entropy = -sum((count/total) * math.log2(count/total) for count in base_counts.values() if count > 0)
    return format_float(entropy, 2)

def calculate_gc_content(sequence: str) -> float:
    """Calculate GC content"""
    if not sequence:
        return 0.0
    # Only consider ATCG bases, filter out 'N', '-', etc.
    valid_bases = [base for base in sequence.upper() if base in 'ATCG']
    if not valid_bases:
        return 0.0
    gc = sum(1 for base in valid_bases if base in ['G', 'C'])
    gc_content = gc / len(valid_bases)
    return format_float(gc_content, 2)

def calculate_continuity_ratio(sequence: str) -> float:
    """Calculate continuity ratio (proportion of consecutive identical bases)"""
    if not sequence or len(sequence) < 2:
        return 0.0
    
    continuity_count = 0
    total_pairs = 0
    
    for i in range(1, len(sequence)):
        # Only consider ATCG bases, filter out 'N', '-', etc.
        if sequence[i].upper() in 'ATCG' and sequence[i-1].upper() in 'ATCG':
            total_pairs += 1
            if sequence[i] == sequence[i-1]:
                continuity_count += 1
    
    continuity_ratio = continuity_count / total_pairs if total_pairs > 0 else 0.0
    return format_float(continuity_ratio, 2)

# ------------------------------
# === Utility Functions: CIGAR Operation Analysis ===
# ------------------------------
def analyze_all_operations(cigartuples):
    """Analyze detailed statistics of all CIGAR operations"""
    if not cigartuples:
        return {
            'total_operations': 0,
            'operations_detail': {},
            'total_bases': 0,
            'large_indels': {'large_ins': [], 'large_del': []},
            'large_indel_summary': {
                'large_ins_count': 0, 'large_del_count': 0,
                'large_ins_max_length': 0, 'large_del_max_length': 0,
                'large_ins_avg_length': 0.0, 'large_del_avg_length': 0.0,
                'large_ins_total_length': 0, 'large_del_total_length': 0
            }
        }
    
    # CIGAR operation code mapping
    cigar_codes = {
        0: 'M', 1: 'I', 2: 'D', 3: 'N', 
        4: 'S', 7: '=', 8: 'X'
    }
    
    # Initialize operation statistics
    op_stats = {
        'total_operations': len(cigartuples),
        'operations_detail': {},
        'total_bases': 0,
        'large_indels': {
            'large_ins': [],  # Store all I operations with length > 30
            'large_del': []   # Store all D operations with length > 30
        }
    }
    
    # Count detailed information for each operation
    for op, length in cigartuples:
        op_code = cigar_codes.get(op, f'Unknown({op})')
        
        if op_code not in op_stats['operations_detail']:
            op_stats['operations_detail'][op_code] = {
                'count': 0,
                'total_length': 0,
                'avg_length': 0.0,
                'max_length': 0,
                'min_length': float('inf')
            }
        
        # Update operation statistics
        stats = op_stats['operations_detail'][op_code]
        stats['count'] += 1
        stats['total_length'] += length
        stats['max_length'] = max(stats['max_length'], length)
        stats['min_length'] = min(stats['min_length'], length)
        
        op_stats['total_bases'] += length
        
        # Count large INDEL operations
        if length >= 30:
            if op == 1:  # Insertion
                op_stats['large_indels']['large_ins'].append(length)
            elif op == 2:  # Deletion
                op_stats['large_indels']['large_del'].append(length)
    
    # Calculate average length (keep 2 decimal places)
    for op_code, stats in op_stats['operations_detail'].items():
        if stats['count'] > 0:
            avg_len = stats['total_length'] / stats['count']
            stats['avg_length'] = format_float(avg_len, 2)
        if stats['min_length'] == float('inf'):
            stats['min_length'] = 0
    
    # Calculate large INDEL statistics (all floating-point numbers keep 2 decimal places)
    large_ins = op_stats['large_indels']['large_ins']
    large_del = op_stats['large_indels']['large_del']
    
    large_ins_avg = format_float(np.mean(large_ins) if large_ins else 0.0, 2)
    large_del_avg = format_float(np.mean(large_del) if large_del else 0.0, 2)
    
    op_stats['large_indel_summary'] = {
        'large_ins_count': len(large_ins),
        'large_del_count': len(large_del),
        'large_ins_max_length': max(large_ins) if large_ins else 0,
        'large_del_max_length': max(large_del) if large_del else 0,
        'large_ins_avg_length': large_ins_avg,
        'large_del_avg_length': large_del_avg,
        'large_ins_total_length': sum(large_ins) if large_ins else 0,
        'large_del_total_length': sum(large_del) if large_del else 0
    }
    
    return op_stats

def safe_float(value, decimals=2):
    """Safely convert to float, handle None and exceptions, keep specified decimal places"""
    if value is None:
        return format_float(0.0, decimals)
    try:
        return format_float(float(value), decimals)
    except (ValueError, TypeError):
        return format_float(0.0, decimals)

def is_padded_sequence(sequence: str) -> bool:
    """Determine if the sequence is pure padded sequence (all N or -)"""
    if not sequence:
        return True
    # Remove all padding characters like N, -, *, etc.
    clean_seq = ''.join([base for base in sequence if base.upper() not in ['N', '-', '*', '?']])
    return len(clean_seq) == 0

def calculate_n_content(sequence: str) -> float:
    """Calculate proportion of N in sequence"""
    if not sequence:
        return 1.0  # Empty sequence considered as all N
    n_count = sum(1 for base in sequence if base.upper() == 'N')
    n_ratio = n_count / len(sequence)
    return format_float(n_ratio, 2)

# ------------------------------
# === Class Definitions ===
# ------------------------------
class PositionMapping:
    """Position mapping information class"""
    
    def __init__(self):
        self.mappings = []  # Store list of (recon_pos, ref_pos, read_pos, operation, base)
    
    def add_mapping(self, recon_pos: int, ref_pos: Optional[int], read_pos: Optional[int], 
                   operation: str, base: str):
        """Add position mapping"""
        self.mappings.append({
            'recon_pos': recon_pos,
            'ref_pos': ref_pos,
            'read_pos': read_pos,
            'operation': operation,
            'base': base
        })
    
    def get_mapping_at_recon_pos(self, recon_pos: int) -> Optional[Dict]:
        """Get information at specified reconstruction position"""
        for mapping in self.mappings:
            if mapping['recon_pos'] == recon_pos:
                return mapping
        return None
    
    def get_mapping_at_ref_pos(self, ref_pos: int) -> List[Dict]:
        """Get information at specified reference position"""
        return [mapping for mapping in self.mappings if mapping['ref_pos'] == ref_pos]
    
    def find_by_ref_pos(self, ref_pos: int) -> Optional[Dict]:
        """Find mapping for specified reference position"""
        for mapping in self.mappings:
            if mapping['ref_pos'] == ref_pos:
                return mapping
        return None

class GlobalReadInfo:
    """Read global information class"""
    
    def __init__(self, read: pysam.AlignedSegment):
        self.read_name = read.query_name
        self.original_length = read.query_length or 0
        self.mapq = read.mapping_quality or 0
        self.nm_tag = read.get_tag('NM') if read.has_tag('NM') else 0
        
        # Alignment completeness weight
        alignment_len = read.query_alignment_length or 0
        self.alignment_completeness = safe_float(alignment_len / self.original_length if self.original_length > 0 else 0.0, 2)
        
        # Sequence complexity features
        self.query_sequence = read.query_sequence or ""
        
        # Check if it's a padded sequence (high N content)
        self.n_content = calculate_n_content(self.query_sequence)
        self.is_padded = self.n_content > 0.9  # Consider as padded if N content > 90%
        
        # Only calculate complexity features for non-padded sequences
        if not self.is_padded:
            self.sequence_complexity = safe_float(calculate_complexity(self.query_sequence), 2)
            self.sequence_entropy = safe_float(calculate_sequence_entropy(self.query_sequence), 2)
            self.gc_content = safe_float(calculate_gc_content(self.query_sequence), 2)
            self.continuity_ratio = safe_float(calculate_continuity_ratio(self.query_sequence), 2)
            repeat_result = calculate_repeat_content(self.query_sequence)
            self.full_repeat_length = repeat_result[0]
            self.full_repeat_ratio = safe_float(repeat_result[1], 2)
        else:
            # Set features to 0 for padded sequences
            self.sequence_complexity = format_float(0.0, 2)
            self.sequence_entropy = format_float(0.0, 2)
            self.gc_content = format_float(0.0, 2)
            self.continuity_ratio = format_float(0.0, 2)
            self.full_repeat_length = 0
            self.full_repeat_ratio = format_float(0.0, 2)
        
        # Detailed statistics of all CIGAR operations
        self.all_operations_stats = analyze_all_operations(read.cigartuples)
        
        # Other basic information
        self.is_reverse = read.is_reverse
        self.is_secondary = read.is_secondary
        self.is_supplementary = read.is_supplementary
        self.reference_start = read.reference_start or 0
        self.reference_end = read.reference_end or 0
        self.cigartuples = read.cigartuples or []
        self.reference_length = max(0, self.reference_end - self.reference_start)
        
        # Keep original read object for reconstruction
        self.read = read

class ReadReconstruction:
    """Read reconstruction class - precise reconstruction according to CIGAR operations"""
    
    def __init__(self, global_read: GlobalReadInfo, ref: pysam.FastaFile, 
                 chrom: str, target_pos: int):
        self.global_info = global_read
        self.chrom = chrom
        self.target_pos = target_pos
        self.target_pos_0based = target_pos - 1
        self.ref = ref
        
        # Reconstructed sequence and position mapping
        self.reconstructed_seq = ""
        self.position_mapping = PositionMapping()
        self.reconstruction_length = 0
        
        # Breakpoint information
        self.breakpoint_info: Optional[Dict] = None
        
        # Left and right sequence information
        self.left_sequence = ""
        self.right_sequence = ""
        
        # Left and right features
        self.left_features = {}
        self.right_features = {}
        
        # CIGAR operation statistics
        self.left_operations = {}
        self.right_operations = {}
        
        # Mark if it's a valid reconstruction (non-padded sequence)
        self.is_valid = False
        
        # Perform reconstruction
        self._reconstruct_alignment(ref)
        if self._calculate_breakpoint_info():
            self._analyze_features()
            self._analyze_cigar_operations_by_region()
            # Check if reconstructed sequence is padded
            n_content = calculate_n_content(self.reconstructed_seq)
            self.is_valid = n_content < 0.9  # Consider valid only if N content < 90%
    
    def _reconstruct_alignment(self, ref: pysam.FastaFile):
        """Precisely reconstruct sequence according to CIGAR operations"""
        read = self.global_info.read
        reconstructed_seq = []
        
        # Initialize pointers
        read_pos = 0  # Position in read sequence
        ref_pos = self.global_info.reference_start  # Position in reference genome
        recon_pos = 0  # Position in reconstructed sequence
        
        # CIGAR operation type mapping
        cigar_ops = {
            0: 'M', 1: 'I', 2: 'D', 3: 'N', 
            4: 'S', 7: '=', 8: 'X'
        }
        
        # Traverse all CIGAR operations
        for op, op_len in self.global_info.cigartuples:
            op_type = cigar_ops.get(op, '?')
            
            if op in (0, 7, 8):  # M, =, X - Match/sequence match/sequence mismatch
                # Get bases from reference genome
                try:
                    ref_bases = ref.fetch(self.chrom, ref_pos, ref_pos + op_len)
                except:
                    # If fetch fails, pad with N
                    ref_bases = 'N' * op_len
                    
                for i in range(op_len):
                    base = ref_bases[i] if i < len(ref_bases) else 'N'
                    reconstructed_seq.append(base)
                    self.position_mapping.add_mapping(
                        recon_pos, ref_pos + i, read_pos + i, op_type, base
                    )
                    recon_pos += 1
                
                read_pos += op_len
                ref_pos += op_len
                
            elif op in (1, 4):  # I (Insertion) or S (Soft clip)
                # Get bases from read sequence
                read_bases = self.global_info.query_sequence[read_pos:read_pos + op_len] if self.global_info.query_sequence else 'N' * op_len
                for i in range(op_len):
                    base = read_bases[i] if i < len(read_bases) else 'N'
                    reconstructed_seq.append(base)
                    self.position_mapping.add_mapping(
                        recon_pos, None, read_pos + i, op_type, base
                    )
                    recon_pos += 1
                
                read_pos += op_len
                # ref_pos does not increase
                
            elif op in (2, 3):  # D (Deletion) or N (Reference skip)
                # Get bases from reference genome
                try:
                    ref_bases = ref.fetch(self.chrom, ref_pos, ref_pos + op_len)
                except:
                    # If fetch fails, pad with N
                    ref_bases = 'N' * op_len
                    
                for i in range(op_len):
                    base = ref_bases[i] if i < len(ref_bases) else 'N'
                    reconstructed_seq.append(base)
                    self.position_mapping.add_mapping(
                        recon_pos, ref_pos + i, None, op_type, base
                    )
                    recon_pos += 1
                
                ref_pos += op_len
                # read_pos does not increase
        
        self.reconstructed_seq = ''.join(reconstructed_seq)
        self.reconstruction_length = len(self.reconstructed_seq)
    
    def _calculate_breakpoint_info(self):
        """Calculate breakpoint information"""
        # Find target position in reconstructed sequence
        mapping = self.position_mapping.find_by_ref_pos(self.target_pos_0based)
        if mapping is None:
            return False
        
        breakpoint_recon_pos = mapping['recon_pos']
        breakpoint_base = mapping['base']
        breakpoint_operation = mapping['operation']
        breakpoint_read_pos = mapping['read_pos']
        
        left_length = breakpoint_recon_pos
        right_length = self.reconstruction_length - breakpoint_recon_pos - 1
        
        # Get left and right sequences
        self.left_sequence = self.reconstructed_seq[:breakpoint_recon_pos]
        self.right_sequence = self.reconstructed_seq[breakpoint_recon_pos + 1:]
        
        # Save breakpoint information
        self.breakpoint_info = {
            'target_position': self.target_pos,
            'breakpoint_index': breakpoint_recon_pos,
            'breakpoint_base': breakpoint_base,
            'breakpoint_operation': breakpoint_operation,
            'breakpoint_read_pos': breakpoint_read_pos if breakpoint_read_pos is not None else 0,
            'left_length': left_length,
            'right_length': right_length,
            'left_read_length': breakpoint_read_pos if breakpoint_read_pos is not None else 0,
            'right_read_length': self.global_info.original_length - (breakpoint_read_pos or 0) - 1,
            'ref_left_length': self.target_pos_0based - self.global_info.reference_start,
            'ref_right_length': self.global_info.reference_end - self.target_pos_0based - 1 if self.global_info.reference_end > self.target_pos_0based else 0
        }
        
        return True
    
    def _analyze_features(self):
        """Analyze left and right sequence features"""
        if not self.breakpoint_info:
            return
        
        # Left features
        left_n_content = calculate_n_content(self.left_sequence)
        left_is_padded = left_n_content > 0.9
        
        if not left_is_padded:
            self.left_features = {
                'sequence': self.left_sequence,
                'length': len(self.left_sequence),
                'read_length': self.breakpoint_info['left_read_length'],
                'ref_length': self.breakpoint_info['ref_left_length'],
                'complexity': safe_float(calculate_complexity(self.left_sequence), 2),
                'entropy': safe_float(calculate_sequence_entropy(self.left_sequence), 2),
                'gc_content': safe_float(calculate_gc_content(self.left_sequence), 2),
                'continuity_ratio': safe_float(calculate_continuity_ratio(self.left_sequence), 2),
                'repeat_length': calculate_repeat_content(self.left_sequence)[0],
                'repeat_ratio': safe_float(calculate_repeat_content(self.left_sequence)[1], 2),
                'n_content': left_n_content,
                'is_padded': False
            }
        else:
            # Set features to 0 for padded sequences
            self.left_features = {
                'sequence': self.left_sequence,
                'length': len(self.left_sequence),
                'read_length': self.breakpoint_info['left_read_length'],
                'ref_length': self.breakpoint_info['ref_left_length'],
                'complexity': format_float(0.0, 2),
                'entropy': format_float(0.0, 2),
                'gc_content': format_float(0.0, 2),
                'continuity_ratio': format_float(0.0, 2),
                'repeat_length': 0,
                'repeat_ratio': format_float(0.0, 2),
                'n_content': left_n_content,
                'is_padded': True
            }
        
        # Right features
        right_n_content = calculate_n_content(self.right_sequence)
        right_is_padded = right_n_content > 0.9
        
        if not right_is_padded:
            self.right_features = {
                'sequence': self.right_sequence,
                'length': len(self.right_sequence),
                'read_length': self.breakpoint_info['right_read_length'],
                'ref_length': self.breakpoint_info['ref_right_length'],
                'complexity': safe_float(calculate_complexity(self.right_sequence), 2),
                'entropy': safe_float(calculate_sequence_entropy(self.right_sequence), 2),
                'gc_content': safe_float(calculate_gc_content(self.right_sequence), 2),
                'continuity_ratio': safe_float(calculate_continuity_ratio(self.right_sequence), 2),
                'repeat_length': calculate_repeat_content(self.right_sequence)[0],
                'repeat_ratio': safe_float(calculate_repeat_content(self.right_sequence)[1], 2),
                'n_content': right_n_content,
                'is_padded': False
            }
        else:
            # Set features to 0 for padded sequences
            self.right_features = {
                'sequence': self.right_sequence,
                'length': len(self.right_sequence),
                'read_length': self.breakpoint_info['right_read_length'],
                'ref_length': self.breakpoint_info['ref_right_length'],
                'complexity': format_float(0.0, 2),
                'entropy': format_float(0.0, 2),
                'gc_content': format_float(0.0, 2),
                'continuity_ratio': format_float(0.0, 2),
                'repeat_length': 0,
                'repeat_ratio': format_float(0.0, 2),
                'n_content': right_n_content,
                'is_padded': True
            }
    
    def _analyze_cigar_operations_by_region(self):
        """Analyze CIGAR operation statistics by left and right regions"""
        if not self.breakpoint_info:
        return
    
    breakpoint_index = self.breakpoint_info['breakpoint_index']
    
    # Initialize operation counters
    left_ops = {'M': 0, 'I': 0, 'D': 0, 'N': 0, 'S': 0, '=': 0, 'X': 0}
    right_ops = {'M': 0, 'I': 0, 'D': 0, 'N': 0, 'S': 0, '=': 0, 'X': 0}
    
    # Count left and right operations
    for mapping in self.position_mapping.mappings:
        if mapping['recon_pos'] < breakpoint_index:
            left_ops[mapping['operation']] += 1
        elif mapping['recon_pos'] > breakpoint_index:
            right_ops[mapping['operation']] += 1
    
    self.left_operations = left_ops
    self.right_operations = right_ops
    
    # Calculate left and right alignment completeness (only for non-padded sequences, keep 2 decimal places)
    if not self.left_features.get('is_padded', True):
        left_effective = sum(count for op, count in left_ops.items() 
                           if op in ['M', '=', 'X', 'D', 'N'])
        left_total = sum(left_ops.values())
        left_completeness = left_effective / left_total if left_total > 0 else 0.0
        self.left_features['alignment_completeness'] = format_float(left_completeness, 2)
    else:
        self.left_features['alignment_completeness'] = format_float(0.0, 2)
    
    if not self.right_features.get('is_padded', True):
        right_effective = sum(count for op, count in right_ops.items() 
                            if op in ['M', '=', 'X', 'D', 'N'])
        right_total = sum(right_ops.values())
        right_completeness = right_effective / right_total if right_total > 0 else 0.0
        self.right_features['alignment_completeness'] = format_float(right_completeness, 2)
    else:
        self.right_features['alignment_completeness'] = format_float(0.0, 2)

@property
def has_breakpoint(self) -> bool:
    return self.breakpoint_info is not None

def get_read_statistics(self) -> Dict:
    """Get all statistical information for this read"""
    if not self.has_breakpoint or not self.is_valid:
        return {}
    
    # Get key metrics from CIGAR statistics
    op_stats = self.global_info.all_operations_stats
    large_stats = op_stats['large_indel_summary']
    
    # Get detailed statistics of key CIGAR operations (ensure all operations have values)
    cigar_details = {}
    for op_code in ['D', 'I', 'M', 'S']:
        if op_code in op_stats['operations_detail']:
            details = op_stats['operations_detail'][op_code]
            cigar_details[f'cigar_{op_code.lower()}_total_length'] = details['total_length'] or 0
            cigar_details[f'cigar_{op_code.lower()}_avg_length'] = format_float(details['avg_length'], 2)
            cigar_details[f'cigar_{op_code.lower()}_max_length'] = details['max_length'] or 0
        else:
            cigar_details[f'cigar_{op_code.lower()}_total_length'] = 0
            cigar_details[f'cigar_{op_code.lower()}_avg_length'] = format_float(0.0, 2)
            cigar_details[f'cigar_{op_code.lower()}_max_length'] = 0
    
    # Ensure all fields have numerical values, no None
    stats = {
        # Basic read information - all values guaranteed to be numerical
        'original_length': self.global_info.original_length,
        'mapq': self.global_info.mapq,
        'alignment_completeness': format_float(self.global_info.alignment_completeness, 2),
        'sequence_complexity': format_float(self.global_info.sequence_complexity, 2),
        'sequence_entropy': format_float(self.global_info.sequence_entropy, 2),
        'gc_content': format_float(self.global_info.gc_content, 2),
        'continuity_ratio': format_float(self.global_info.continuity_ratio, 2),
        'full_repeat_length': self.global_info.full_repeat_length,
        'full_repeat_ratio': format_float(self.global_info.full_repeat_ratio, 2),
        
        # Length statistics
        'reference_coverage_length': self.global_info.reference_length,
        'reconstruction_length': self.reconstruction_length,
        
        # CIGAR operation statistics
        'cigar_total_operations': op_stats['total_operations'],
        'cigar_total_bases': op_stats['total_bases'],
        
        # Detailed CIGAR operation statistics
        **cigar_details,
        
        # Large INDEL statistics
        'large_ins_total_length': large_stats['large_ins_total_length'],
        'large_ins_avg_length': format_float(large_stats['large_ins_avg_length'], 2),
        'large_ins_max_length': large_stats['large_ins_max_length'],
        'large_del_total_length': large_stats['large_del_total_length'],
        'large_del_avg_length': format_float(large_stats['large_del_avg_length'], 2),
        'large_del_max_length': large_stats['large_del_max_length'],
        
        # Breakpoint left statistics
        'left_recon_length': self.left_features.get('length', 0),
        'left_read_length': self.left_features.get('read_length', 0),
        'left_ref_length': self.left_features.get('ref_length', 0),
        'left_alignment_completeness': format_float(self.left_features.get('alignment_completeness', 0.0), 2),
        'left_complexity': format_float(self.left_features.get('complexity', 0.0), 2),
        'left_entropy': format_float(self.left_features.get('entropy', 0.0), 2),
        'left_gc_content': format_float(self.left_features.get('gc_content', 0.0), 2),
        'left_continuity_ratio': format_float(self.left_features.get('continuity_ratio', 0.0), 2),
        'left_repeat_length': self.left_features.get('repeat_length', 0),
        'left_repeat_ratio': format_float(self.left_features.get('repeat_ratio', 0.0), 2),
        
        # Breakpoint right statistics
        'right_recon_length': self.right_features.get('length', 0),
        'right_read_length': self.right_features.get('read_length', 0),
        'right_ref_length': self.right_features.get('ref_length', 0),
        'right_alignment_completeness': format_float(self.right_features.get('alignment_completeness', 0.0), 2),
        'right_complexity': format_float(self.right_features.get('complexity', 0.0), 2),
        'right_entropy': format_float(self.right_features.get('entropy', 0.0), 2),
        'right_gc_content': format_float(self.right_features.get('gc_content', 0.0), 2),
        'right_continuity_ratio': format_float(self.right_features.get('continuity_ratio', 0.0), 2),
        'right_repeat_length': self.right_features.get('repeat_length', 0),
        'right_repeat_ratio': format_float(self.right_features.get('repeat_ratio', 0.0), 2),
        
        # Validity markers
        'is_valid': True,
        'left_is_padded': self.left_features.get('is_padded', True),
        'right_is_padded': self.right_features.get('is_padded', True)
    }
    
    return stats

# ------------------------------
# === Main Analysis Function ===
# ------------------------------
def analyze_sv_sites(txt_file, bam_file, ref_file, output_dir, extend_length, select_read, csv_path):
    """Analyze SV sites and generate CSV statistical file"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Read input file
    try:
        if txt_file.endswith('.txt'):
            df = pd.read_csv(txt_file, sep='\t', header=0, dtype=str)
        else:
            df = pd.read_csv(txt_file, header=0, dtype=str)
    except Exception as e:
        print(f"[ERROR] Cannot read input file {txt_file}: {e}")
        return
    
    print(f"[INFO] Loaded {len(df)} SV sites")
    
    # Collect all statistical results
    all_results = []
    
    for idx, row in df.iterrows():
        chrom = str(row['chr']).strip()
        pos = int(float(row['pos']))  # Handle possible floating-point numbers
        svtype = str(row.get('sv_type', 'NA')).strip()
        svlen = str(row.get('sv_len', 'NA')).strip()
        identifier = f"{chrom}.{pos}.{svtype}.{svlen}"
        print(f"[INFO] Analyzing position {idx+1}/{len(df)}: {identifier}")
        
        # Open BAM and reference files
        try:
            bam = pysam.AlignmentFile(bam_file, "rb")
            ref = pysam.FastaFile(ref_file)
        except Exception as e:
            print(f"  [ERROR] Cannot open BAM or reference file: {e}")
            continue
        
        # Collect all reads for this position
        reads = []
        read_count = 0
        try:
            for read in bam.fetch(chrom, pos-1, pos):
                if read.is_unmapped or read.mapping_quality < 10:
                    continue
                
                # Limit number of reads
                if select_read and len(reads) >= select_read:
                    break
                
                try:
                    # Create global read information
                    global_read = GlobalReadInfo(read)
                    # Create reconstruction information
                    reconstruction = ReadReconstruction(global_read, ref, chrom, pos)
                    if reconstruction.has_breakpoint:
                        reads.append(reconstruction)
                    read_count += 1
                except Exception as e:
                    # print(f"  Warning: Error processing read {read.query_name}: {e}")
                    continue
        except Exception as e:
            print(f"  [ERROR] Error reading BAM file: {e}")
        
        bam.close()
        ref.close()
        
        print(f"  Scanned {read_count} reads, found {len(reads)} reads containing breakpoint")
        
        # Filter out valid reads (non-padded sequences)
        valid_reads = [r for r in reads if r.is_valid]
        padded_reads = [r for r in reads if not r.is_valid]
        
        print(f"  Among them valid reads: {len(valid_reads)}, padded reads: {len(padded_reads)}")
        
        if not valid_reads:
            # If no valid reads, add empty row
            result_row = {
                'id': identifier,
                'avg_gc': format_float(0.0, 2),
                'avg_continuity': format_float(0.0, 2),
                'avg_entropy': format_float(0.0, 2),
                'num_reads_used': 0,
                'num_valid_reads': 0,
                'num_padded_reads': len(padded_reads),
                'total_candidates': read_count
            }
            all_results.append(result_row)
            continue
        
        # Calculate statistical values for all valid reads
        all_stats = []
        for read in valid_reads:
            stats = read.get_read_statistics()
            # Ensure all values are numerical
            cleaned_stats = {k: safe_float(v, 2) for k, v in stats.items()}
            all_stats.append(cleaned_stats)
        
        # Calculate averages - only using valid reads, keep 2 decimal places
        avg_stats = {}
        if all_stats:
            for key in all_stats[0].keys():
                values = [s[key] for s in all_stats if key in s]
                # Filter out None values
                numeric_values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
                if numeric_values:
                    avg_value = np.mean(numeric_values)
                    avg_stats[key] = format_float(avg_value, 2)
                else:
                    avg_stats[key] = format_float(0.0, 2)
        
        # Calculate traditional GC, continuity, entropy - only using left+right sequences of valid reads
        all_left_gc = []
        all_right_gc = []
        all_left_cont = []
        all_right_cont = []
        all_left_ent = []
        all_right_ent = []
        
        for read in valid_reads:
            if not read.left_features.get('is_padded', True):
                all_left_gc.append(safe_float(read.left_features.get('gc_content', 0.0), 2))
                all_left_cont.append(safe_float(read.left_features.get('continuity_ratio', 0.0), 2))
                all_left_ent.append(safe_float(read.left_features.get('entropy', 0.0), 2))
            
            if not read.right_features.get('is_padded', True):
                all_right_gc.append(safe_float(read.right_features.get('gc_content', 0.0), 2))
                all_right_cont.append(safe_float(read.right_features.get('continuity_ratio', 0.0), 2))
                all_right_ent.append(safe_float(read.right_features.get('entropy', 0.0), 2))
        
        # Combine left and right data
        all_gc = all_left_gc + all_right_gc
        all_cont = all_left_cont + all_right_cont
        all_ent = all_left_ent + all_right_ent
        
        # Calculate averages, keep 2 decimal places
        numeric_gc = [v for v in all_gc if v is not None and not math.isnan(v)]
        avg_gc = format_float(np.mean(numeric_gc) if numeric_gc else 0.0, 2)
        
        numeric_cont = [v for v in all_cont if v is not None and not math.isnan(v)]
        avg_cont = format_float(np.mean(numeric_cont) if numeric_cont else 0.0, 2)
        
        numeric_ent = [v for v in all_ent if v is not None and not math.isnan(v)]
        avg_ent = format_float(np.mean(numeric_ent) if numeric_ent else 0.0, 2)
        
        # Build result row
        result_row = {
            'id': identifier,
            'avg_gc': avg_gc,
            'avg_continuity': avg_cont,
            'avg_entropy': avg_ent,
            'num_reads_used': len(reads),  # Total reads (including padded)
            'num_valid_reads': len(valid_reads),  # Valid reads count
            'num_padded_reads': len(padded_reads),  # Padded reads count
            'total_candidates': read_count
        }
        
        # Add all average statistics (only based on valid reads, all keep 2 decimal places)
        for key, value in avg_stats.items():
            result_row[f'avg_{key}'] = format_float(value, 2)
        
        all_results.append(result_row)
    
    # Build DataFrame and save
    if all_results:
        # Extract all possible columns
        all_columns = set()
        for row in all_results:
            all_columns.update(row.keys())
        
        # Fixed column order
        base_columns = ['id', 'avg_gc', 'avg_continuity', 'avg_entropy', 
                       'num_reads_used', 'num_valid_reads', 'num_padded_reads', 'total_candidates']
        
        # Other columns in alphabetical order
        other_columns = sorted([col for col in all_columns if col not in base_columns])
        column_order = base_columns + other_columns
        
        result_df = pd.DataFrame(all_results, columns=column_order)
        
        # Ensure all numerical columns are floats and keep 2 decimal places
        for col in result_df.columns:
            if col != 'id' and col != 'total_candidates' and col != 'num_reads_used' and col != 'num_valid_reads' and col != 'num_padded_reads':
                # Try to convert to float and keep 2 decimal places
                try:
                    result_df[col] = result_df[col].apply(lambda x: format_float(x, 2) if pd.notnull(x) else 0.0)
                except:
                    pass
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
        result_df.to_csv(csv_path, index=False)
        print(f"[INFO] Statistical CSV saved to: {csv_path}")
        print(f"[INFO] Analyzed {len(all_results)} sites, generated {len(result_df.columns)} statistical columns")
        print(f"[INFO] All floating-point numbers unified to 2 decimal places")
        
        # Display first few columns as example
        sample_cols = result_df.columns.tolist()[:15]
        print(f"[INFO] Example column names: {', '.join(sample_cols)}...")
    else:
        print("[WARNING] No results generated")

# ------------------------------
# === Main Function ===
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="SV Site Statistical Analysis Tool")
    parser.add_argument('--txt_file', required=True, help="Input txt/csv file containing chr,pos,sv_type,sv_len columns")
    parser.add_argument('--bam_file', required=True, help="Input BAM file")
    parser.add_argument('--ref_file', required=True, help="Reference genome FASTA file")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('--extend_length', type=int, default=500, help="Extension length (for identifier generation)")
    parser.add_argument('--select_read', type=int, default=30, help="Maximum number of reads to analyze per site")
    parser.add_argument('--csv_out', default='sv_statistics.csv', help="Output CSV filename")
    
    args = parser.parse_args()
    
    # Build complete CSV path
    csv_path = args.csv_out
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(args.output_dir, csv_path)
    
    # Run analysis
    analyze_sv_sites(
        txt_file=args.txt_file,
        bam_file=args.bam_file,
        ref_file=args.ref_file,
        output_dir=args.output_dir,
        extend_length=args.extend_length,
        select_read=args.select_read,
        csv_path=csv_path
    )

if __name__ == "__main__":
    main()
