#!/usr/bin/env python3
"""
Create metadata.jsonl file for all images in the images directory
"""

import json
import os
import argparse
from pathlib import Path

def create_metadata_file(images_dir: str, output_file: str):
    """
    Create metadata.jsonl file for all images in the directory
    
    Args:
        images_dir: Directory containing image files
        output_file: Output metadata.jsonl file path
    """
    # Supported image formats
    supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    
    # Get all image files
    image_files = []
    for file in os.listdir(images_dir):
        if any(file.lower().endswith(fmt) for fmt in supported_formats):
            image_files.append(file)
    
    # Sort files for consistent ordering
    image_files.sort()
    
    print(f"Found {len(image_files)} image files in {images_dir}")
    
    # Create metadata entries
    metadata_entries = []
    for image_file in image_files:
        entry = {
            "file_name": f"images/{image_file}",
            "text": ""
        }
        metadata_entries.append(entry)
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in metadata_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Created metadata file: {output_file}")
    print(f"Total entries: {len(metadata_entries)}")
    
    # Show first few entries
    print("\nFirst 5 entries:")
    for i, entry in enumerate(metadata_entries[:5]):
        print(f"  {i+1}. {entry['file_name']}")

def main():
    parser = argparse.ArgumentParser(
        description='Create metadata.jsonl file for all images in directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_metadata.py -i /path/to/images -o metadata.jsonl
  python create_metadata.py --images-dir /path/to/images --output metadata.jsonl
        """
    )
    
    parser.add_argument(
        '-i', '--images-dir',
        type=str,
        default="/home/ubuntu/musinsaigo/02_preproces-image/train_data/proc_dataset/images",
        help='Directory containing image files (default: /home/ubuntu/musinsaigo/02_preproces-image/train_data/proc_dataset/images)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default="/home/ubuntu/musinsaigo/02_preproces-image/train_data/proc_dataset/metadata_all.jsonl",
        help='Output metadata.jsonl file path (default: /home/ubuntu/musinsaigo/02_preproces-image/train_data/proc_dataset/metadata_all.jsonl)'
    )
    
    args = parser.parse_args()
    
    # Validate images directory
    if not os.path.exists(args.images_dir):
        print(f"❌ Images directory not found: {args.images_dir}")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    try:
        create_metadata_file(args.images_dir, args.output)
        print(f"\n✅ Successfully created metadata file with all images!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error creating metadata file: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 