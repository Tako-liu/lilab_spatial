import argparse
import csv
import numpy as np
import cv2
import time
import tifffile
import pandas as pd
import matplotlib.pyplot as plt
import getopt
import sys
import os
#import utils
import cellpose
import cellpose.utils
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap


def add_legend(src_img, small_legend_img, base_rate=30):
    small_legend_img = small_legend_img[..., :3]  # 避免透明度通道报错
    img_height, img_weight = src_img.shape[:2]
    small_legend_height, small_legend_weight = small_legend_img.shape[:2]

    max_rate = img_height / small_legend_height
    if base_rate > max_rate:
        base_rate = max_rate

    new_legend_height = img_height
    new_legend_weight = int(small_legend_weight * base_rate)
    tmp_legend = cv2.resize(small_legend_img, (new_legend_weight, int(small_legend_height * base_rate)))
    # 新建一个黑底等高的图像
    # new_legend = np.zeros((new_legend_height, new_legend_weight, 3), np.uint8)
    new_legend = np.ones((new_legend_height, new_legend_weight, 3), np.uint8) * background_color
    new_legend = new_legend.astype(np.uint8)
    # 居中摆放
    height_start = (new_legend.shape[0] - tmp_legend.shape[0]) // 2
    new_legend[height_start: height_start+tmp_legend.shape[0], :tmp_legend.shape[1], ...] = tmp_legend

    # 间隔像素
    split_pixes = new_legend.shape[1] // 4
    split_img = np.ones((new_legend_height, split_pixes, 3), np.uint8) * background_color
    split_img = split_img.astype(np.uint8)

    return cv2.hconcat([src_img, split_img, new_legend])


def draw_color_bar(cmap, max_value, save_path):
    fig, ax = plt.subplots(1, 1)
    if background_color != [0, 0, 0]:
        colorbar_background_color = background_color / np.array(255)
        ax.figure.set_facecolor(colorbar_background_color)  # 设置背景色
    fraction = 1  # .05
    norm = mpl.colors.Normalize(vmin=0, vmax=max_value)
    cbar = ax.figure.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                              ax=ax, pad=.05, extend='neither', fraction=fraction) #extend='both'
    if background_color == [0, 0, 0]:
        colorbar_background_color = background_color / np.array(255)
        ax.figure.set_facecolor(colorbar_background_color)  # 设置背景色
        # 设置colorbar的标签字体颜色
        #cbar.ax.yaxis.label.set_color('red')  # 将标签颜色设置为红色s
        for label in cbar.ax.get_yticklabels():
            label.set_color('white')  # 将刻度标签颜色设置为白色x
        # 设置颜色条的刻度线颜色h
        cbar.ax.tick_params(axis="y", color="white",pad=1) #pad标签与刻度线的位置
    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight')


def draw_umi_img(gene_ori, cells, save_dir, c_map='RdYlBu_r', background_color=(0, 0, 0)):

    cells_type = [col[0] for col in gene_ori if col[1] != 0]
    cells_type = [int(str(tmp_cell).lstrip("cell_")) for tmp_cell in cells_type]

    umi_values = np.asarray([float(col[1]) for col in gene_ori if col[1] != 0])

    cell_color = np.ones((cells.max() + 1, 3), np.uint8) * background_color
    cell_color = cell_color.astype(np.uint8)
    # 依据umi count值生成颜色值
    max_value = umi_values.max()
    if len(c_map) < 14:
        print(c_map)
        cn = mpl.colormaps.get_cmap(c_map)
    else:
        c_map=c_map.split(",")
        cn = mcolors.LinearSegmentedColormap.from_list("mycmap", c_map)
        print("#color")
    #cn = plt.cm.get_cmap(c_map)
    cn_colors = np.array(cn(umi_values/max_value)[:, [0, 1, 2]] * 255, dtype='int')
    # cn_colors = np.array(cn(np.array(df['umi'].values)//STEP)[:, [0, 1, 2]] * 255, dtype='int')
    # 画colorbar
    draw_color_bar(cn, max_value, os.path.join(save_dir, name+"_legend.tif"))
    # 基础颜色为cmap(0)的颜色，merge之后可能全白
    cell_color[1:, ...] = (255 * np.asarray(cn(0)[0:3])).astype(int)
    # 如果注释掉上面这句,改成下面的，则基础颜色为黑色，即不表达的基因的细胞为黑色
    # cell_color[1:, ...] = np.array([0, 0, 0], dtype=int)

    cluster_dict = {cell: umi_color for cell, umi_color in zip(cells_type, cn_colors)}

    for i in range(len(cell_color)):
        if i in cluster_dict:
            cell_color[i] = cluster_dict[i]

    # print("save img...")
    # 画可视化图像
    return cell_color[cells]
#定义一个颜色转换的函数
class ColorConverter:
    """
    提供HEX和RGB之间的转换功能。
    """
    @staticmethod
    def hex_to_rgb(hex_color):
        """
        将HEX颜色转换为RGB颜色。        
        :param hex_color: HEX格式的颜色代码，如'#ff0000'。
        :return: 对应的RGB元组，如(255, 0, 0)。
        """
        return tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in range(0, 6, 2))    
    @staticmethod
    def rgb_to_hex(rgb_color):
        """
        将RGB颜色转换为HEX颜色。       
        :param rgb_color: RGB格式的颜色代码，如(255, 0, 0)。
        :return: 对应的HEX字符串，如'#ff0000'。
        """
        return '#' + ''.join([hex(c)[2:].rjust(2, '0') for c in rgb_color])
 



if __name__ == "__main__":
    # main()

    print(sys.argv)
    parser = argparse.ArgumentParser(description="Draw gene umi images")
    parser.add_argument('--csv', type=str, help="input cvs, colum: cell_id, umi_value", required=True)
    parser.add_argument('--outdir', type=str, help="output dir", required=True)
    parser.add_argument('--cmap', type=str, help="color_map ,like 'RdYlBu_r'", required=True)
    parser.add_argument('--cells_npy', type=str, help="cells.npy path", required=True)
    parser.add_argument('--name', help="figure name", type=str, default="umi")
    parser.add_argument('--background_color', type=str, default="#000000",
                        help="If give, background color will change.eg:'#000000'")
    parser.add_argument('--line_color', type=str, default="#000000",
                    help="If give, background color will change.eg:'#000000'")                    
    args = parser.parse_args()

    cells_npy = args.cells_npy
    gene_path = args.csv
    save_dir = args.outdir
    cmap = args.cmap
    name = args.name
    background_color = args.background_color
    background_color = [int(background_color[1:3], 16), int(background_color[3:5], 16), int(background_color[5:7], 16)]
    t0 = time.time()
    line_color = args.background_color
    line_color = ColorConverter.hex_to_rgb(line_color)

    with open(gene_path, 'r') as f:
        gene_csv = csv.reader(f)
        gene_ori = list(gene_csv)

    gene_ori = gene_ori[1:]  # 丢掉第一行无效信息

    cells = np.load(cells_npy)

    cell_color_img = draw_umi_img(gene_ori, cells, save_dir, c_map=cmap, background_color=background_color)
    print("Draw cost time is {}".format(time.time() - t0))

    # 添加边界线
    outline = cellpose.utils.masks_to_outlines(cells)
    #cell_color_img[outline] = [0, 0, 0]
    cell_color_img[outline] = line_color

    tifffile.imwrite(os.path.join(save_dir, name+".tif"), cell_color_img, compression="jpeg")

    # 添加legend
    legend = tifffile.imread(os.path.join(save_dir, name+"_legend.tif"))
    umi_he_img = add_legend(cell_color_img, legend)
    tifffile.imwrite(os.path.join(save_dir, name+"_with_legend.tif"), umi_he_img, compression="jpeg")
