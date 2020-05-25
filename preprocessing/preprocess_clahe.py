import cv2
import os
import glob
from tqdm import tqdm

BASE_DIR = os.path.abspath('/mnt/dr_data/cropped')
OUT_DIR = os.path.abspath('/mnt/dr_data/clahe')
IMG_SIZE = 800
images_paths = [os.path.basename(path) for path in glob.glob(BASE_DIR + '/*')]

def load_img(filename):
    img = cv2.imread(os.path.join(BASE_DIR, filename))
    return img

def resize(img):
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_AREA)

def apply_clahe(img, grid_size = 5):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    lab_planes = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=1.5,tileGridSize=(grid_size,grid_size))
    lab_planes[0] = clahe.apply(lab_planes[0])
    img = cv2.merge(lab_planes)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    return img

def process_img(img_name):
    img = load_img(img_name)
    img = apply_clahe(img)   
    img = resize(img)
    return img

for img_path in tqdm(images_paths):
    try:
        img = process_img(img_path)
        cv2.imwrite(os.path.join(OUT_DIR, img_path), img)
    except KeyboardInterrupt:
        print('You cancelled the operation.')
        break
    except:
        print('The error occured during processing image:', img_path)