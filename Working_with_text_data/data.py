import re 
import os 
import requests

def get_data(file_path):
    if not os.path.exists(file_path):
        URL = (
            "https://raw.githubusercontent.com/rasbt/"
            "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
            "the-verdict.txt"
        )
        
        response = requests.get(URL, timeout = 30)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)

if __name__ == '__main__':
    file_path = "Working_with_text_data/the-verdict.txt"
    get_data(file_path)

    