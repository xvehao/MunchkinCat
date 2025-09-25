from PIL import Image
import scanpy as sc
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def image_crop(adata, img, crop_size=224, 
               target_size=224, save_path=None, 
               verbose=False):
    tile_names = []

    with tqdm(total=len(adata), desc='Tiling Image', bar_format='{l_bar}{bar} [ time left: {remaining} ]') as pbar:
        for image_row, image_col, barcode in zip(adata.obs['x_pixel'], adata.obs['y_pixel'], adata.obs_names):
            image_down = image_row - crop_size / 2
            image_up = image_row + crop_size / 2
            image_left = image_col - crop_size / 2
            image_right = image_col + crop_size / 2

            tile = img.crop(
                (image_left, image_down, image_right, image_up)
            )
            tile.thumbnail((target_size, target_size), Image.LANCZOS)
            tile.resize((target_size, target_size))
            tile_name = barcode # str(image_col) + '-' + str(image_row) + '-' + str(crop_size)
            if save_path is not None:
                out_tile = Path(save_path) / (tile_name + '.png')
                tile_names.append(str(out_tile))
                if verbose:
                    print('Generating tile at location ({}, {})'.format(str(image_col), str(image_row)))
                tile.save(out_tile, 'PNG')
            pbar.update(1)

    adata.obs['slice_path'] = tile_names
    return adata


def main():
    Image.MAX_IMAGE_PIXELS = None

    metadata = pd.read_csv("SEDR_analyses/data/BRCA1/metadata.tsv", sep='\t')
    dir_input = "SEDR_analyses/data/BRCA1/V1_Human_Breast_Cancer_Block_A_Section_1"
    img = Image.open('./SEDR_analyses/data/BRCA1/V1_Human_Breast_Cancer_Block_A_Section_1/V1_Breast_Cancer_Block_A_Section_1_image.tif')

    adata = sc.read_10x_h5(f'{dir_input}/filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()

    spatial=pd.read_csv(f"{dir_input}/spatial/tissue_positions_list.csv",sep=",",header=None,na_filter=False,index_col=0)

    adata.obs["x1"]=spatial[1]
    adata.obs["x2"]=spatial[2]
    adata.obs["x3"]=spatial[3]
    adata.obs["x4"]=spatial[4]
    adata.obs["x5"]=spatial[5]

    adata=adata[adata.obs["x1"]==1]
    adata.var_names=[i.upper() for i in list(adata.var_names)]
    adata.var["genename"]=adata.var.index.astype("str")
    adata.obs['pred'] = metadata['annot_type'].values

    adata.obs["x_array"]=adata.obs["x2"]
    adata.obs["y_array"]=adata.obs["x3"]
    adata.obs["x_pixel"]=adata.obs["x4"]
    adata.obs["y_pixel"]=adata.obs["x5"]

    image_crop(adata, img, crop_size=224, target_size=224, save_path="D:/hw/RuiPath/10xBRC", verbose=True)

if __name__=='__main__':
    main()