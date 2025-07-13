# Copyright IBM Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT


import json
import math
from pathlib import Path
from zipfile import ZipFile
import os
import deepsearch as ds
import typer
from PIL import Image
import fitz
import io
import tqdm
from typing import List, Union, Dict
import glob
import pandas as pd
# from docling.document_converter import DocumentConverter


    
ds_key = "ENTER YOUR KEY HERE"

def load_description_from_image_path(img_path: str) -> str:
    ret_str = ''
    text_path = img_path.replace('png', 'txt')
    if os.path.isfile(text_path):
        with open(text_path, 'rt') as text_f:
            ret_str = text_f.read()
    
    return ret_str

    
def extract_figures_from_json_doc(
    pdf_filename: Path, document: dict, output_dir: Path, resolution: int = 72
):
    """
    Iterate through the converted document format and extract the figures as PNG files

    Parameters
    ----------
    pdf_filename : Path
        Input PDF file.
    document :
        The converted document from Deep Search.
    bbox : List[int]
        Bounding box to extract, in the format [x0, y0, x1, y1], where the origin is the top-left corner.
    output_dir : Path
        Output directory where all extracted images will be saved.
    resolution : int
        Resolution of the extracted image.
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
        
    output_base = output_dir / document["file-info"]["filename"].rstrip(".pdf").rstrip(
        ".PDF"
    )
    page_counters = {}
    doc = fitz.open(pdf_filename)
    # Iterate through all the figures identified in the converted document
    for figure in document.get("figures", []):
        prov = figure["prov"][0]
        page = prov["page"]
        page_counters.setdefault(page, 0)
        page_counters[page] += 1

        # Retrieve the page dimensions, needed for shifting the coordinates of the bounding boxes
        page_dims = next(
            (dims for dims in document["page-dimensions"] if dims["page"] == page), None
        )
        if page_dims is None:
            typer.secho(
                f"Page dimensions for page {page} not defined! Skipping it.",
                fg=typer.colors.YELLOW,
            )
            continue

        # Convert the Deep Search bounding box in the coordinate frame used to extract images.
        # From having the origin in the bottom-left corner, to the top-left corner
        # The bounding box is expanded to the closest integer coordinates, because of the format
        # requirements of the tools used in the extraction.
        bbox = [
            math.floor(prov["bbox"][0]),
            math.floor(page_dims["height"] - prov["bbox"][3]),
            math.ceil(prov["bbox"][2]),
            math.ceil(page_dims["height"] - prov["bbox"][1]),
        ]

        # Extract the bounding box
        output_filename = output_base.with_name(
            f"{output_base.name}_{page}_{page_counters[page]}"
        )
        dpi = resolution   
        scale = dpi / 72
        bbox = [b * scale for b in bbox]
        
        page = doc.load_page(page - 1)
        pix = page.get_pixmap(dpi=dpi)
        image_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(image_bytes))        
        crop = image.crop(bbox)        
        crop.save(f'{output_filename}.png')
        if figure['text']:
            with open(f'{output_filename}.txt', 'wt') as caption_f:
                caption_f.write(figure['text'])
        
        typer.secho(f"Figure extracted in {output_filename}.png", fg=typer.colors.GREEN)        
        print(f"Figure extracted in {output_filename}.png")        


def extract_tables_from_json_doc(pdf_filename: Path, document: dict, output_dir: Path, resolution:int = 300):
    """
    Iterate through the converted document format and extract the figures as PNG files

    Parameters
    ----------
    pdf_filename : Path
        Input PDF file.
    document :
        The converted document from Deep Search.
    output_dir : Path
        Output directory where all extracted images will be saved.
    """

    output_base = output_dir / document["file-info"]["filename"].rstrip(".pdf").rstrip(
        ".PDF"
    )
    page_counters = {}
    doc = fitz.open(pdf_filename)
    # Iterate through all the tables identified in the converted document
    for table in document.get("tables", []):
        prov = table["prov"][0]
        page = prov["page"]
        page_counters.setdefault(page, 0)
        page_counters[page] += 1

        # Load the table into a Pandas DataFrame
        table_content = [[cell["text"] for cell in row] for row in table["data"]]
        df = pd.DataFrame(table_content)
        
        if 'text' in table:
            tab_caption = table['text']            
            output_filename = output_base.with_name(
                f"{output_base.name}_{page}_{page_counters[page]}_caption.cap"
        )
            with open(output_filename, "wt") as save_f:
                save_f.write(tab_caption)
            typer.secho(f"Table caption saved in {output_filename}", fg=typer.colors.RED)
            
        # Save table
        output_filename = output_base.with_name(
            f"{output_base.name}_{page}_{page_counters[page]}.csv"
        )
        df.to_csv(output_filename, index=False)
        typer.secho(f"Table extracted in {output_filename}", fg=typer.colors.GREEN)
        
        # Extract the table image    
        # -----------------------    
        # Retrieve the page dimensions, needed for shifting the coordinates of the bounding boxes
        page_dims = next(
            (dims for dims in document["page-dimensions"] if dims["page"] == page), None
        )
        if page_dims is None:
            typer.secho(
                f"Page dimensions for page {page} not defined! Skipping it.",
                fg=typer.colors.YELLOW,
            )
            continue
        
        bbox = [
            math.floor(prov["bbox"][0]),
            math.floor(page_dims["height"] - prov["bbox"][3]),
            math.ceil(prov["bbox"][2]),
            math.ceil(page_dims["height"] - prov["bbox"][1]),
        ]

        # Extract the bounding box
        output_filename = output_base.with_name(
            f"{output_base.name}_{page}_{page_counters[page]}_table"
        )
        dpi = resolution   
        scale = dpi / 72
        bbox = [b * scale for b in bbox]
        
        page = doc.load_page(page - 1)
        pix = page.get_pixmap(dpi=dpi)
        image_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(image_bytes))        
        crop = image.crop(bbox)        
        crop.save(f'{output_filename}.png')
        
        typer.secho(f"Table image extracted in {output_filename}", fg=typer.colors.GREEN)


def init_ds_client():
    profile_name = "livexiv"
    api = ds.CpsApi.from_env(profile_name=profile_name)
    return api


def load_documents(api, pdf_filename: str, output_dir: str, proj_key: str ):
    # Launch the docucment conversion and download the results\
    documents = ds.convert_documents(
        api=api, proj_key=proj_key, source_path=pdf_filename, progress_bar=True
    )
    
    documents.download_all(result_dir=output_dir, progress_bar=True)


def extract_figures(pdf_filename:str, output_dir: str, dpi: int):
    # Iterate through the zip files which were downloaded and loop through the content of each zip archive
    for output_file in Path(output_dir).rglob("*.legacy.json"):
        # print(str(output_file) + '\n\n')
        with open(str(output_file), "r") as json_f:
            document = json.load(json_f)
        extract_figures_from_json_doc(
            pdf_filename, document, output_dir, resolution=dpi
        )
        
        os.remove(output_file)


def extract_tables(pdf_filename:str, output_dir: str, dpi: int):
    # Iterate through the zip files which were downloaded and loop through the content of each zip archive
    for output_file in Path(output_dir).rglob("*.legacy.json"):
        with open(str(output_file), "r") as json_f:
                document = json.load(json_f)
        extract_tables_from_json_doc(
                        Path(pdf_filename), document, Path(output_dir), resolution=dpi
                    )
        
        os.remove(output_file)                    
                    
                                    
def extract_figures_from_pdf(pdf_filename: str, output_dir: str, dpi: int = 300):
    api = init_ds_client()
    load_documents(api, pdf_filename, output_dir, ds_key)
    extract_figures(pdf_filename, output_dir, dpi)
    
        
def extract_tables_from_pdf(pdf_filename: str, output_dir: str, dpi: int = 300):
    api = init_ds_client()
    load_documents(api, pdf_filename, output_dir, ds_key)
    extract_tables(pdf_filename, output_dir, dpi)
    

    
def get_all_paragraphs_and_captions(pdf_filename:str, output_dir: str) -> List[Union[List[str], List[str]]]:
    # Iterate through the zip files which were downloaded and loop through the content of each zip archive
    for output_file in Path(output_dir).rglob("json*.zip"):
        with ZipFile(output_file) as archive:
            all_files = archive.namelist()
            for name in all_files:
                if name.endswith(".json"):
                    typer.secho(
                        f"Procecssing file {name} in archive {output_file}",
                        fg=typer.colors.BLUE,
                    )
                    document = json.loads(archive.read(name))
                    par_list = []
                    for p in document['main-text']:
                        if p['type'] == 'paragraph' and p['name'] == 'text':
                            par_list.append(p['text'])                        
                            
                    cap_list = []
                    for p in document['figures']:
                        if p['text']:
                            cap_list.append(p['text'])                    
    return par_list, cap_list     



def get_all_img_cap_pairs_above_length_th(src_dir: str, length_th: int) -> Dict:            
    cap_img_pairs = {}      
    img_paths = sorted(glob.glob(src_dir + '/*.png'))
    for img_path in tqdm.tqdm(img_paths, total=len(img_paths), desc='searching for detailed captions'):
        text_desc = load_description_from_image_path(img_path)
        if not text_desc:            
            continue
        if len(text_desc) >= length_th:
            cap_img_pairs[Path(img_path).stem] = {'img_path': os.path.abspath(img_path), 'caption': text_desc}
    
    return cap_img_pairs


def extract_pars_and_caps_from_pdf(pdf_filename: str, output_dir: str) -> List[Union[List[str], List[str]]]:
    api = init_ds_client()
    load_documents(api, pdf_filename, output_dir, ds_key)    
    paragraph_list, captions_list = get_all_paragraphs_and_captions(pdf_filename, output_dir)
    return paragraph_list, captions_list


