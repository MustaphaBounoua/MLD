import os
import cv2
import glob
import numpy as np
from PIL import Image

# Taken and Updated from https://github.com/switchablenorms/CelebAMask-HQ/

def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


#list1
#label_list = ['skin', 'neck', 'hat', 'eye_g', 'hair', 'ear_r', 'neck_l', 'cloth', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'nose', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip']
#list2 
# label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

#Without skin mask

def create_mask(color_type="gray"):
    color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], 
                    [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], 
                    [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], 
                    [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], 
                    [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]

    label_list = ['nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
    label_list2 = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

    folder_base = "/home/bounoua/work/mld/data/data_celba/CelebAMask-HQ/CelebAMask-HQ-mask-anno"
    img_num = 30000

    if color_type == "gray":
        folder_save = "/home/bounoua/work/mld/data/data_celba/CelebAMask-HQ/CelebAMaskHQ-mask/"
        make_folder(folder_save)
        for k in range(img_num):
            folder_num = k // 2000
            im_base = np.zeros((512, 512))
            for idx, label in enumerate(label_list):
                filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
                if (os.path.exists(filename)):
                    print (label, idx+1)
                    im = cv2.imread(filename)
                    im = im[:, :, 0]
                    im_base[im != 0] = 255

            filename_save = os.path.join(folder_save, str(k) + '.png')
            print (filename_save)
            cv2.imwrite(filename_save, im_base)
    else:
        folder_base = "/home/bounoua/work/mld/data/data_celba/CelebAMask-HQ/CelebAMask-HQ-mask-temp"
      #  folder_base = "./data/CelebAMask-HQ/CelebAMaskHQ-mask-temp"
        folder_save = "/home/bounoua/work/mld/data/data_celba/CelebAMask-HQ/CelebAMaskHQ-mask-color"
        #folder_save = "./data/CelebAMask-HQ/CelebAMaskHQ-mask-color"
        make_folder(folder_save)

        for k in range(img_num):
            filename = os.path.join(folder_base, str(k) + '.png')
            if (os.path.exists(filename)):
                im_base = np.zeros((512, 512,3))
                im = Image.open(filename)
                im = np.array(im)
                for idx, color in enumerate(color_list):
                    # print (color, idx)
                    im_base[im == idx] = color

            filename_save = os.path.join(folder_save, str(k) + '.png')
            result = Image.fromarray((im_base).astype(np.uint8))
            print (filename_save)
            result.save(filename_save)

create_mask()