import os
import arxiv
import requests
from lxml import etree
from lxml import html
import logging
from pathlib import Path
import time
logger = logging.getLogger('livexiv')


def get_tex_source(arxiv_obj, query: str, download_path: str):
    for lidx in range(len(arxiv_obj.links)):
        if arxiv_obj.links[lidx].title is None:  # abstract has no title in arxiv.result class
            arxiv_url = arxiv_obj.links[lidx].href
            break

    arxiv_src_link = arxiv_url.replace('abs', 'src')
    file_stamp = arxiv_src_link.split('/')[-1]
    filename = f'{file_stamp}_{query}.tar.gz'
    filepath = os.path.join(download_path, filename)
    response = requests.get(arxiv_src_link)
    open(filepath, "wb").write(response.content)
    
    
def download_arxiv_papers(download_path='./artifacts', verbose=True, max_results: int = 10, download_src: bool = False):
    # Construct the default API client.
    client = arxiv.Client() 
    queries = ['cs.AI', 'cs.CV', 'cs.LG', 'cs.RO', 'q-bio.BM', 'eess.SP', 'eess.SY', 'q-bio.GN', 'q-bio.CB', 'q-bio.TO', 
               'physics.optics', 'physics.bio-ph', 'physics.app-ph', 'physics.data-an'][:1]
    pdf_paths = []
    entries = []
    if download_src:
        tex_src_path = os.path.join(download_path, 'text_src')
        os.makedirs(tex_src_path, exist_ok=True)
        
    for query in queries:
        search = arxiv.Search(
        query = query,
        max_results = 150,
        sort_by = arxiv.SortCriterion.SubmittedDate
        )
        cnt = 0
        results = client.results(search)
        time.sleep(5)
        for r in results:            
            try:
                if chcek_if_paper_has_license(r):
                    logger.info(f'{r.title} has a license - skipping')
                else:
                    if verbose:
                        logger.info(f'{query}: {r.title}')
                    
                    id = r.entry_id.split('/')[-1]                
                    if id not in entries:
                        retry_cnt = 0
                        if download_src:
                            get_tex_source(r, query, tex_src_path)
                            
                        r.download_pdf(dirpath=download_path, filename=f"{id}_{query}.pdf")                
                        pdf_paths.append(os.path.join(download_path, f"{id}_{query}.pdf"))
                        entries.append(id)
                        cnt += 1
            except Exception as e:
                logging.warning(e)
                continue
                    
            if cnt == max_results:
                break     
                      
    return pdf_paths


def chcek_if_paper_has_license(arxiv_obj) -> bool:
    for lidx in range(len(arxiv_obj.links)):
        if arxiv_obj.links[lidx].title is None:  # abstract has no title in arxiv.result class
            arxiv_url = arxiv_obj.links[lidx].href
            break
        
    response = requests.get(arxiv_url)
    html_content = response.content  
    
    parser = html.HTMLParser()
    tree = etree.HTML(html_content, parser)

    has_license = tree.xpath('//*[@class="has_license"]')
    return has_license

