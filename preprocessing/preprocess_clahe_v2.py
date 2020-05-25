import cv2
import os
import glob
from tqdm import tqdm
from pathlib import Path
import numpy as np
import sys

BASE_DIR = os.path.abspath('/mnt/dr_data/train')
OUT_DIR = os.path.abspath('/mnt/dr_data/clahe_v2')
images_paths = [os.path.basename(path) for path in glob.glob(BASE_DIR + '/*')]

def load_img(filename):
    img = cv2.imread(os.path.join(BASE_DIR, filename))
    return img

def load_img(filename):
    img = cv2.imread(BASE_DIR + '/' + filename)
    return img

def resize_image_aspect_ratio(img, new_width=None, new_height=None):
    height, width = img.shape[:2]
    if new_width is not None and new_height is None:
        r = new_width/width
        new_height = int(height * r)
    elif new_width is None and new_height is not None:
        r = new_height/height
        new_width = int(width * r)
    new_image = cv2.resize(img, (new_width, new_height))
    return new_image

def resize(img):
    return cv2.resize(img, (600, 600), interpolation = cv2.INTER_AREA)

def crop(img, r, c):
    y_min = max(c[0] - r, 0)
    y_max = min(c[0] + r, img.shape[0])
    x_min = max(c[1] - r, 0)
    x_max = min(c[1] + r, img.shape[1])
    # print(y_min , y_max, x_min , x_max)
    return img[y_min : y_max, x_min : x_max]

def calc_radius_and_center(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    y_center = img.shape[0] // 2
    y_offset = 50
    x_center = img.shape[1] // 2
    x_offset = 50

    m_l = img[y_center-y_offset:y_center+y_offset, :x_offset].mean() * 1.1
    thsh_l = np.argwhere(img[y_center-y_offset:y_center+y_offset, :x_center] <= m_l)[:,1]

    m_r = img[y_center-y_offset:y_center+y_offset, -x_offset:].mean() * 1.1
    thsh_r = np.argwhere(img[y_center-y_offset:y_center+y_offset, x_center:] <= m_r)[:,1]
    thsh_r = thsh_r + x_center

    r = (thsh_r.min() - thsh_l.max()) // 2
    if r < 1:
        return x_center, (y_center, x_center)
    else:
        return r, (y_center, thsh_l.max() + r)

def fill_everything_out_of_radius(img, r, center):
    mask = np.full((img.shape[0], img.shape[1], 3), 0, dtype=np.uint8)
    cv2.circle(mask, (center[1], center[0]), r, (1,1,1), -1, 8, 0)
    img = img*mask + 128*(1-mask)
    return img.astype(np.uint8)

def apply_clahe(img, grid_size = 5):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    lab_planes = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=1.5,tileGridSize=(grid_size,grid_size))
    lab_planes[0] = clahe.apply(lab_planes[0])
    img = cv2.merge(lab_planes)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    return img

def normalize(img):
    normalizedImg = np.zeros(img.shape)
    normalizedImg = cv2.normalize(img,  normalizedImg, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return normalizedImg

def process_img(img_name):
    img = load_img(img_name)
    img = resize_image_aspect_ratio(img, new_height=900)
    img = normalize(img)
    r, center = calc_radius_and_center(img)
    img = apply_clahe(img)
    r = int(r * 0.95)
    img = fill_everything_out_of_radius(img, r, center)
    img = crop(img, r, center)    
    img = resize(img)
    return img

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
for img_path in tqdm(images_paths):
    try:
        img = process_img(img_path)
        cv2.imwrite(os.path.join(OUT_DIR, img_path), img)
    except KeyboardInterrupt:
        print('You cancelled the operation.')
        break
    except:
        print('The error occured during processing image:', img_path, '.', sys.exc_info()[0])