#!/usr/bin/env python
"""
Log Cleanup Script

Clean up and organize old experiment logs.
"""

import shutil
from pathlib import Path
from datetime import datetime
import argparse


def cleanup_logs(logs_dir: Path, archive_dir: Path, keep_recent: int = 5):
    """Clean up old log directories
    
    Args:
        logs_dir: Directory containing log folders
        archive_dir: Directory to move old logs to
        keep_recent: Number of recent runs to keep
    """
    if not logs_dir.exists():
        print(f"Logs directory {logs_dir} does not exist")
        return
    
    # Get all batch run directories
    batch_dirs = [d for d in logs_dir.iterdir() 
                  if d.is_dir() and d.name.startswith('batch_run_')]
    
    # Sort by creation time (newest first)
    batch_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"Found {len(batch_dirs)} batch run directories")
    
    if len(batch_dirs) <= keep_recent:
        print(f"Keeping all {len(batch_dirs)} directories (≤ {keep_recent})")
        return
    
    # Directories to archive
    to_archive = batch_dirs[keep_recent:]
    
    print(f"Keeping {keep_recent} recent directories:")
    for i, d in enumerate(batch_dirs[:keep_recent]):
        size_mb = sum(f.stat().st_size for f in d.rglob('*') if f.is_file()) / 1024 / 1024
        print(f"  {i+1}. {d.name} ({size_mb:.1f} MB)")
    
    # Create archive directory
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nArchiving {len(to_archive)} old directories to {archive_dir}:")
    
    total_size = 0
    for d in to_archive:
        size_mb = sum(f.stat().st_size for f in d.rglob('*') if f.is_file()) / 1024 / 1024
        total_size += size_mb
        
        print(f"  📦 {d.name} ({size_mb:.1f} MB)")
        
        # Move to archive
        archive_path = archive_dir / d.name
        if archive_path.exists():
            shutil.rmtree(archive_path)
        shutil.move(str(d), str(archive_path))
    
    print(f"\n✅ Archived {len(to_archive)} directories ({total_size:.1f} MB total)")


def main():
    parser = argparse.ArgumentParser(description="Clean up LLMEvo experiment logs")
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"),
                        help="Directory containing logs")
    parser.add_argument("--archive-dir", type=Path, default=Path("archived_logs"), 
                        help="Directory to move old logs to")
    parser.add_argument("--keep-recent", type=int, default=5,
                        help="Number of recent runs to keep")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without actually doing it")
    
    args = parser.parse_args()
    
    print("🧹 LLMEvo Log Cleanup")
    print("=" * 40)
    print(f"Logs directory: {args.logs_dir}")
    print(f"Archive directory: {args.archive_dir}")
    print(f"Keep recent: {args.keep_recent}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be moved")
        print()
    
    cleanup_logs(args.logs_dir, args.archive_dir, args.keep_recent)


if __name__ == "__main__":
    main()