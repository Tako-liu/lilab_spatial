import argparse
from scipy import ndimage as ndi
import numpy as np
import tifffile
import cv2
import os
import sys


if __name__ == '__main__':
    # cells_path = r"D:\code\cell-seg\img\img_dist_fg4\cells.npy"
    # save_dir = r"D:\code\cell-seg\img\img_dist_fg4\tmp"
    # he_path = r"D:\code\cell-seg\img\img_dist_fg4\he.tif"

    print(sys.argv)
    parser = argparse.ArgumentParser(description="Find all cells center loc, and show it on black image"
                                                 " or the img given")

    parser.add_argument('--cells_path', type=str, help="cells.npy or nucleus.npy path", required=True)
    parser.add_argument('--out_path', type=str, help="out image path", required=True)
    parser.add_argument('--img_path', type=str, help="HE or FL image ", required=False)
    args = parser.parse_args()

    cells_path = args.cells_path
    save_dir = args.out_path
    img_path = args.img_path

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cell_loc = []
    masks = np.load(cells_path)
    if img_path is None:
        img = np.zeros((*masks.shape[:2], 3), np.uint8)
    else:
        img = tifffile.imread(img_path)
        if len(img.shape) == 2:  # gray image
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(img, masks.shape[:2][::-1])

    slices = ndi.find_objects(masks.astype(int))
    for i, si in enumerate(slices):
        if si is not None:
            sr, sc = si
            mask = (masks[sr, sc] == (i + 1)).astype(np.uint8)
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pvc, pvr = np.concatenate(contours[-2], axis=0).squeeze().T
            vr, vc = pvr + sr.start, pvc + sc.start
            if hasattr(vr, '__iter__'):
                center = (int(vc.mean()), int(vr.mean()))
                cv2.circle(img, center, 3, (255, 0, 0), -1)
                img[vr, vc] = (0, 255, 0)
                #cell_loc.append("\t".join([str(i+1), str(center[0]), str(center[1])]))
                cell_loc.append("\t".join(["cell_" + str(i+1), str(center[0]), str(center[1])]))
            else:
                '''
                这里会因为细胞识别会出现只有1个像素的情况，无法创建可以迭代的列表，
                1个像素太少了可以认为是误检，直接丢弃
                '''
                print()
                print(vc)
                print(vr)
                continue

            if i % 10000 == 0:
                print(i)

    with open(os.path.join(save_dir, "cells_center.txt"), "w") as f:
        f.write("\n".join(cell_loc))
    tifffile.imwrite(os.path.join(save_dir, "cells_center.tif"), img, compression="jpeg")
