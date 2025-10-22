#!/usr/bin/env python3
"""
Process spatial transcriptomics data from HEST format.
Extracts gene expression data for patches and saves processed results.
"""

import argparse
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import anndata as ad


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process spatial transcriptomics data from hest_data/st to comply with the input of scfoundation and match with barcodes of hest_st/patches'
    )
    parser.add_argument(
        '--mycodepath',
        type=str,
        required=True,
        default='/mnt/DATA-4/hx/Ruipath/MunchkinCat/pretrain/',
        help='Path to MunchkinCat/pretrain/'
    )
    parser.add_argument(
        '--scfpath',
        type=str,
        required=True,
        default='/mnt/DATA-4/hx/Ruipath/scFoundation/model/',
        help='Path to scFoundation model directory'
    )
    parser.add_argument(
        '--hestpath',
        type=str,
        required=True,
        default='/mnt/DATA-4/hx/Ruipath/hest_data/',
        help='Path to HEST data directory'
    )
    parser.add_argument(
        '--gene-list',
        type=str,
        default=None,
        help='Path to gene list TSV file (default: scfpath/OS_scRNA_gene_index.19264.tsv)'
    )
    
    return parser.parse_args()


def load_gene_list(gene_list_path):
    """Load gene list from TSV file."""
    gene_list_df = pd.read_csv(gene_list_path, header=0, delimiter='\t')
    return list(gene_list_df['gene_name'])


def process_slide(slide_id, hestpath, gene_list, main_gene_selection):
    """
    Process a single slide: extract patches and corresponding gene expression.
    
    Parameters
    ----------
    slide_id : str
        Slide identifier
    hestpath : Path
        Path to HEST data directory
    gene_list : list
        List of genes to select
    main_gene_selection : function
        Gene selection function from imported module
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Load patch data and barcodes
        h5path = hestpath / "patches" / f"{slide_id}.h5"
        with h5py.File(h5path, 'r') as f:
            imgs = f['img'][:]
            barcodes = f['barcode'][:].flatten().astype(str)
        
        print(f"  Loaded {len(barcodes)} patches for {slide_id}")
        
        # Load spatial transcriptomics data
        st_path = hestpath / "st" / f"{slide_id}.h5ad"
        adata = ad.read_h5ad(st_path)
        
        # Subset to barcodes from patches
        new_adata = adata[barcodes]
        
        # Verify alignment
        assert new_adata.n_obs == imgs.shape[0], \
            f"Mismatch: {new_adata.n_obs} cells vs {imgs.shape[0]} patches"
        
        # Convert to DataFrame for gene selection
        X_df = pd.DataFrame(new_adata.X.toarray() if hasattr(new_adata.X, 'toarray') else new_adata.X,
                           columns=new_adata.var_names)
        
        # Apply gene selection
        X_df, _ = main_gene_selection(X_df, gene_list)
        
        print(f"  Selected {X_df.shape[1]} genes from {len(gene_list)} gene list")
        
        # Save processed data
        output_dir = hestpath / "processed_st"
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / f"{slide_id}.h5"
        with h5py.File(output_path, "w") as f:
            f.create_dataset("st_processed", 
                           data=X_df.values.astype(np.float32), 
                           dtype='float32')
            f.create_dataset("barcode",
                             data=X_df.values.astype(np.float32), 
                             dtype='float32')
        
        print(f"  ✓ Saved to {output_path}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {slide_id}: {str(e)}")
        return False


def main():
    """Main processing function."""
    # Parse arguments
    args = parse_arguments()
    
    # Convert paths to Path objects
    mycodepath = Path(args.mycodepath)
    scfpath = Path(args.scfpath)
    hestpath = Path(args.hestpath)
    
    # Validate paths
    for path, name in [(mycodepath, 'mycodepath'), 
                       (scfpath, 'scfpath'), 
                       (hestpath, 'hestpath')]:
        if not path.exists():
            print(f"Error: {name} does not exist: {path}")
            sys.exit(1)
    
    # Add custom code paths to sys.path
    sys.path.append(str(mycodepath))
    sys.path.append(str(scfpath))
    
    # Import custom modules
    try:
        from load import main_gene_selection
        from data_loader import Get_hest_meta
        print("✓ Successfully imported custom modules")
    except ImportError as e:
        print(f"Error importing custom modules: {e}")
        sys.exit(1)
    
    # Load gene list
    gene_list_path = args.gene_list
    if gene_list_path is None:
        gene_list_path = scfpath / "OS_scRNA_gene_index.19264.tsv"
    
    gene_list_path = Path(gene_list_path)
    if not gene_list_path.exists():
        print(f"Error: Gene list file not found: {gene_list_path}")
        sys.exit(1)
    
    gene_list = load_gene_list(gene_list_path)
    print(f"✓ Loaded {len(gene_list)} genes from {gene_list_path}")
    
    # Get all slide IDs
    slide_ids, _ = Get_hest_meta()
    print(f"\nProcessing {len(slide_ids)} slides...")
    
    # Process each slide
    success_count = 0
    for i, slide_id in enumerate(slide_ids, 1):
        print(f"\n[{i}/{len(slide_ids)}] Processing {slide_id}...")
        if process_slide(slide_id, hestpath, gene_list, main_gene_selection):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}/{len(slide_ids)} slides")
    if success_count < len(slide_ids):
        print(f"Failed: {len(slide_ids) - success_count} slides")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()