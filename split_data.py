#!/usr/bin/env python3
import os
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def classify_one_file(png_path: Path, src_anchor: Path, dst_root: Path):
    """
    png_path: Full path of the original PNG file
    src_anchor: Source-side "parent of root" path, used to preserve the 03.CLR70 level
    dst_root: Output root directory
    """
    # Key: Preserve the 03.CLR70 level
    rel_dir = png_path.parent.relative_to(src_anchor)
    out_base = dst_root / rel_dir

    ins_dir   = out_base / "Ins_positive"
    del_dir   = out_base / "Del_positive"
    match_dir = out_base / "Match_negative"

    ins_dir.mkdir(parents=True, exist_ok=True)
    del_dir.mkdir(parents=True, exist_ok=True)
    match_dir.mkdir(parents=True, exist_ok=True)

    name = png_path.name
    if "INS" in name:
        shutil.copy2(png_path, ins_dir / name)
    elif "DEL" in name:
        shutil.copy2(png_path, del_dir / name)
    elif "MATCH" in name:
        shutil.copy2(png_path, match_dir / name)
    else:
        shutil.copy2(png_path, match_dir / name)   # Default to Match_negative

def classify_all_folders(src_root: str, dst_root: str, max_workers: int = 8):
    src_path = Path(src_root).resolve()
    dst_path = Path(dst_root).resolve()

    # ****** Change 1: Move reference directory up one level to preserve 03.CLR70 ******
    src_anchor = src_path.parent

    all_pngs = list(src_path.rglob("*.png"))
    if not all_pngs:
        print("No .png files found")
        return

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(classify_one_file, f, src_anchor, dst_path) for f in all_pngs]
        for fut in as_completed(futures):
            fut.result()

    print(f"Completed! Processed {len(all_pngs)} images → {dst_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively traverse all subdirectories in source folder for .png files, "
                    "classify by INS/DEL/MATCH and preserve 03.CLR70 level"
    )
    parser.add_argument("source_dir", help="Original image root directory, e.g., /.../03.CLR70")
    parser.add_argument("target_root", help="Output root directory")
    parser.add_argument("-t", "--threads", type=int, default=8, help="Number of concurrent threads")
    args = parser.parse_args()

    classify_all_folders(args.source_dir, args.target_root, args.threads)
