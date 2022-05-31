import requests
import os
from tqdm.auto import tqdm
import zipfile
import argparse 


def getargs():

    p = argparse.ArgumentParser()
    p.add_argument('glove_url', type=str, help='Path to the glove vectors, this should not change', default="https://nlp.stanford.edu/data/glove.6B.zip")
    p.add_argument('root_folder', type=str, help="Folder for the vectors to be downloaded, may need to be absolute path", default='./')
    p=p.parse_args()
    return p

def checkFolderPath(folder_path):

    if os.path.exists(folder_path):
        print(f'{folder_path} found')
    else:
        print(f'{folder_path} does not exist\nCreating {folder_path}')
        os.mkdir(folder_path)
        

def downloadGloveVectors(url, root_folder):

    filename = url.split('/')[-1]
    vector_folder = f'{root_folder}/vectors/'
    
    checkFolderPath(vector_folder)
    
    filepath = f'{vector_folder}{filename}'
    response = requests.get(url, stream=True)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc='Downloading vectors')

    with open(filepath, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

    else:
        print(f'Success!\nGlove Vectors saved to {filepath}\nHave a nice day')
    
    return filepath, vector_folder


def unzipFile(vector_path, zip_path):

    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(vector_path)
            print(f"Extracted all files to {vector_path}")
            print('\nContents of {vector_path}: ')
            
            for i in os.listdir(vector_path):
                print(i)
            
    except:
        print("Invalid file")


def open_vectors(glove_path):
    
    with open(glove_path, 'rb') as file:

        vectors = file.read().splitlines()
        
    return vectors

def process_vectors(raw_vectors):
    
    vocab = {}

    for vec in tqdm(raw_vectors, total=len(raw_vectors)):
        
        splat_vec = vec.decode().split(' ')
        word = splat_vec[0]
        vector = np.array(splat_vec[1:], dtype=float)
        vocab[word] = vector

    return vocab


def download_glove(url, root_folder):

    zip_path, vector_folder = downloadGloveVectors(url, root_folder)
    unzipFile(vector_folder, zip_path)
    return 'Success!'

if __name__ == '__main__':

    args = getargs()
    print(args.glove_url)
    download_glove(args.glove_url, args.root_folder)
