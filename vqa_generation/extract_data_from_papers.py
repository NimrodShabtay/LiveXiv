import os
from arxiv_utils import download_arxiv_papers
from document_utils import  extract_figures_from_pdf, load_description_from_image_path, extract_tables_from_pdf
import pickle
import glob
from pathlib import Path
import logging
import filecmp
logger = logging.getLogger('livexiv')


def check_overlap_with_prev_version(prev_ver_dir: str, cur_ver_dir: str):
    if prev_ver_dir:
        dirs_cmp = filecmp.dircmp(prev_ver_dir, cur_ver_dir)
        for f in dirs_cmp.common_files:
            if f.endswith('pdf'):
                full_path = os.path.join(cur_ver_dir, f)
                os.remove(full_path)
                logger.info(f'{f} already exsits in the previous version ({prev_ver_dir}) - deleting')

    
def extract_papers_and_data_for_generation(args):
    os.makedirs(args.artifacts_dir, exist_ok=True)
    # ArXiv download
    existing_pdf_paths = glob.glob(args.artifacts_dir + '/*.pdf')
    if not args.generate.external_paper_list:
        if args.arxiv.do_download or not existing_pdf_paths:  # Download on empty folder or append to folder with papers
            pdf_paths = download_arxiv_papers(download_path=args.artifacts_dir, max_results=args.arxiv.max_results, download_src=args.arxiv.download_src)
        else:
            pdf_paths = existing_pdf_paths
    else:
        import pickle
        with open(args.generate.external_paper_list, 'rb') as f:
            external_pdf_list = pickle.load(f)
            
        pdf_paths = external_pdf_list
    
    check_overlap_with_prev_version(args.arxiv.prev_version_dir, args.artifacts_dir)
    pdf_paths = glob.glob(args.artifacts_dir + '/*.pdf')
    logger.info(f'{len(pdf_paths)} arxiv papers downloaded')
    
    if args.assests_type in ['all', 'figures']:
        figures_path = os.path.join(args.artifacts_dir, 'figures')
        os.makedirs(figures_path, exist_ok=True)
        for pdf_path in pdf_paths:
            try:            
                paper_name = Path(pdf_path).stem
                paper_name = paper_name[:-1] if paper_name.endswith('SP') else paper_name  # workaround for extracting image removing last P (happens when loading external pdf list)                
                img_candidates = glob.glob(figures_path + '/*.png')
                img_candidates = [f for f in img_candidates if 'table' not in f]
                existing_images_names = [Path(p).stem for p in img_candidates]
                if any([True if paper_name in img_name else False for img_name in existing_images_names]):  # Skip extracting image if already exsited (assuming extraction is not stopped in the middle)
                    continue 
                else:
                    extract_figures_from_pdf(pdf_path, output_dir = figures_path, dpi=args.arxiv.dpi)
            except Exception as e:
                logger.warning(e)
                continue
    if args.assests_type in ['all', 'tables']:
        tables_path = os.path.join(args.artifacts_dir, 'tables')
        os.makedirs(tables_path, exist_ok=True)
        for pdf_path in pdf_paths:
            try:            
                paper_name = Path(pdf_path).stem
                paper_name = paper_name[:-1] if paper_name.endswith('SP') else paper_name  # workaround for extracting image removing last P                
                existing_images_names = [Path(p).stem for p in glob.glob(tables_path + '/*_table.png')]
                if any([True if paper_name in img_name else False for img_name in existing_images_names]):  # Skip extracting image if already exsited (assuming extraction is not stopped in the middle)                
                    continue 
                else:                    
                    extract_tables_from_pdf(pdf_path, output_dir = tables_path, dpi=args.arxiv.dpi)
            except Exception as e:
                logger.warning(e)
                continue