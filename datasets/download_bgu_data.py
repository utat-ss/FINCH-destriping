import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time


def locate_mat_files(url):
    response = requests.get(url)

    if response.status_code == 200:
        #parse the HTML content for .mat files
        soup = BeautifulSoup(response.text, 'html.parser')
        mat_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.mat')]
        
        #removing all duplicates
        mat_links = list(set(mat_links))
        print(f"Found {len(mat_links)} .mat files")
        # print(mat_links)
        
    else:
        print(f"Failed to retrieve the webpage. status_code: {response.status_code}")
        exit(2)
    
    return mat_links


def download(mat_links, output_dir):
    counter = 0
    total_files = len(mat_links)
    
    for mat_link in mat_links:
        start_time = time.time()
        file_name = os.path.join(output_dir, os.path.basename(mat_link))
        file_response = requests.get(mat_link)

        #save file
        with open(file_name, 'wb') as file:
            file.write(file_response.content)
        end_time = time.time()
        
        counter += 1
        
        print(f"Downloaded: {file_name}")
        print(f"Total Data cubes downloaded: {counter}")
        print(f"Estrimated Remaining time: {(end_time - start_time)*(total_files - counter)/60} minutes")
        


if __name__ == "__main__":
    website_url = "https://icvl.cs.bgu.ac.il/hyperspectral/"
    mat_links = locate_mat_files(website_url)
    
    #create output dir
    output_dir = "more_hyperspectral_data"
    os.makedirs(output_dir, exist_ok=True)

    download(mat_links, output_dir)
