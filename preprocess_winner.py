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

def scaleRadius(img, scale):
    x = img[img.shape[0] // 2, :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img,(0,0),fx=s,fy=s)

def winner_processing(img):
    scale = 300    
    #scaleimgtoagivenradius
    a = scaleRadius(img, scale)
    #subtractlocalmeancolor
    a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0,0), scale/30), -4, 128)
    #removeouter10%
    b = np.zeros(a.shape)
    cv2.circle(b, (a.shape[1] // 2, a.shape[0] // 2), int(scale*0.9),(1,1,1),-1,8,0)
    a = a*b + 128*(1-b)
    return a.astype(np.uint8)

def process_img(img_name):
    img = load_img(img_name)
    img = winner_processing(img)
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