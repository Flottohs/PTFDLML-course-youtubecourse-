

import requests
import os
from pathlib import Path
data_path = Path('/Users/haydenfletcher/Documents/programming/books-course/PTFDLML/notes/files/data')

custom_image_path = data_path / '04-pizza-dad.jpeg'

#download from github

if not custom_image_path.is_file():
    
    with open(custom_image_path,'wb') as f:
        request = requests.get('https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/images')
        
        print(f'downloading{custom_image_path}')
        f.write(request.content)
else: 
    print(f'custom image:{custom_image_path} already downlaoded, skipping')
    print(f"The image is located at: {custom_image_path.resolve()}")