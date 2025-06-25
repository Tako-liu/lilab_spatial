import os
import sys
import cv2
import numpy as np
import tifffile
import csv
import argparse
import utils


def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)  # XXX assume int() truncates!
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q


def draw_legend(legend_color_dict):
    legend_pic = np.zeros((len(legend_color_dict)*100, 500, 3), dtype=np.uint8)
    for i, (cluster, color) in enumerate(legend_color_dict.items()):
        color = np.asarray(color, dtype=int)
        start_loc = np.asarray([30, (i+1) * 100 - 50])
        cv2.circle(legend_pic, start_loc, 20, tuple([int(x) for x in color]), -1)
        cv2.putText(legend_pic, cluster, start_loc+[30, 13], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    thickness=2)
    return legend_pic


def add_legend(src_img, small_legend_img, base_rate=12):
    img_height, img_weight = src_img.shape[:2]
    small_legend_height, small_legend_weight = small_legend_img.shape[:2]

    max_rate = img_height / small_legend_height
    if base_rate > max_rate:
        base_rate = max_rate

    new_legend_height = img_height
    new_legend_weight = int(small_legend_weight * base_rate)
    tmp_legend = cv2.resize(small_legend_img, (new_legend_weight, int(small_legend_height * base_rate)))
    # 新建一个黑底等高的图像
    new_legend = np.zeros((new_legend_height, new_legend_weight, 3), np.uint8)
    # 居中摆放
    height_start = (new_legend.shape[0] - tmp_legend.shape[0]) // 2
    new_legend[height_start: height_start+tmp_legend.shape[0], :tmp_legend.shape[1], ...] = tmp_legend

    # 间隔像素
    split_pixes = new_legend.shape[1] // 4

    return cv2.hconcat([src_img, np.zeros((new_legend_height, split_pixes, 3), np.uint8), new_legend])


if __name__ == '__main__':
    print(sys.argv)
    parser = argparse.ArgumentParser(description="Draw cluster images")

    parser.add_argument('--cells_path', type=str, help="cells.npy path", required=True)
    parser.add_argument('--cluster_path', type=str, help="cluster csv path", required=True)
    parser.add_argument('--save_dir', type=str, help="save dir", required=True)
    parser.add_argument('--redo', action="store_true", default=False, help="Recover color info or not")
    parser.add_argument('--cluster_and_color', type=str, default="",
                        help="If give, only draw cluster which has given."
                             " eg.:'1,2' or given color '1:#FF0000, 2:#00FF00'")
    args = parser.parse_args()

    redo = args.redo
    cluster_path = args.cluster_path
    cells_path = args.cells_path
    save_dir = args.save_dir
    cluster_and_color = args.cluster_and_color

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print("cells path:{}\ncluster path:{}\nsave dir:{}\nredo:{}\ncluster_and_color{}".format(cells_path, cluster_path,
                                                                                             save_dir, redo,
                                                                                             cluster_and_color))
    # 指定输出cluster
    special_cluster_dict = {}
    if cluster_and_color != "":
        cluster_and_color = [special_cluster.strip() for special_cluster in cluster_and_color.split(",")
                             if special_cluster.strip() != ""]

        for special_cluster in cluster_and_color:
            special_cluster = special_cluster.split(":")
            if len(special_cluster) == 1:
                special_cluster_dict[special_cluster[0]] = None
            elif len(special_cluster) == 2:
                color_RGB = special_cluster[1]
                color_RGB = [int(color_RGB[1:3], 16), int(color_RGB[3:5], 16), int(color_RGB[5:7], 16)]
                special_cluster_dict[special_cluster[0]] = color_RGB
            else:
                raise Exception("cluster_and_color get wrong input!")

    cells = np.load(cells_path)

    with open(cluster_path, 'r') as f:
        cluster_csv = csv.reader(f)
        clusters_ori = list(cluster_csv)

    #  传入的clusters 第一列是细胞编号，第二列是cluseter注释， 第三列自己生成，根据注释种类生成0， 1， 2....
    clusters_ori = clusters_ori[1:]
    clusters_ori.sort(key=lambda x: len(x[-1]))  # 按照注释排序一下，固定一下顺序
    if special_cluster_dict != {}:  # 指定展示模式
        clusters_ori = [tmp_cluster for tmp_cluster in clusters_ori if tmp_cluster[1] in special_cluster_dict]

    legend = {}   # cluster注释对应颜色序号
    clusters = []
    for clu in clusters_ori:
        if clu[1] not in legend:
            legend[clu[1]] = len(legend)
        clusters.append([int(clu[0].lstrip('cell_')), str(clu[1]), legend[clu[1]]])  # 丢掉"cell_"前缀

    # clusters = np.asarray(clusters)
    cluster_dict = {x[0]: x[2] for x in clusters}

    type_num = len(legend)
    if os.path.exists(os.path.join(save_dir, "clusters_colors.npy")) and not redo:
        colors = np.load(os.path.join(save_dir, "clusters_colors.npy"))
        print("clusters_colors.npy is exists, load it")
    else:
        # colors = np.asarray([hsv_to_rgb(np.random.rand()*0.5+0.5, np.random.rand()*0.5+0.5, 1.0) for x in range(type_num)])
        colors = np.asarray([hsv_to_rgb(np.random.rand(), np.random.rand(), np.random.rand()*0.5+0.5) for x in range(type_num)])
        colors = (colors*255).astype(np.uint8)
        try:
            col_ = ["#F56867","#FEB915","#C798EE","#59BE86","#7495D3","#D1D1D1","#6D1A9C","#15821E","#3A84E6","#70e014",
                    "#787878","#DB4C6C","#0430e0","#554236","#AF5F3C","#ff7700","#e00417","#DAB370","#fcfc05","#268785",
                    "#ed1299","#09f9f5","#246b93","#cc8e12","#d561dd","#c93f00","#ddd53e","#4aef7b","#e86502","#9ed84e",
                    "#39ba30","#6ad157","#8249aa","#99db27","#e07233","#ff523f","#ce2523","#f7aa5d","#cebb10","#03827f",
                    "#931635","#373bbf","#a1ce4c","#ef3bb6","#d66551","#1a918f","#ff66fc","#2927c4","#7149af","#57e559",
                    "#8e3af4","#f9a270","#22547f","#db5e92","#edd05e","#6f25e8","#0dbc21","#280f7a","#6373ed","#5b910f",
                    "#7b34c1","#0cf29a","#d80fc1","#51f59b"]
            for i in range(len(colors)):
                tmp_col = col_[i]
                colors[i] = [int(tmp_col[1:3], 16), int(tmp_col[3:5], 16), int(tmp_col[5:7], 16)]

            # cell_color = {'Leydig': [55, 50, 72], 'Spermatocyte': [28, 190, 84], 'Elongating': [176, 72, 11],
            #               'RoundSpermatid': [176, 24, 0], 'Sertoli': [1, 105, 178], 'Endothelial': [58, 35, 4],
            #               'InnateLymphoid': [151, 194, 101], 'Myoid': [190, 102, 238], 'Macrophage': [192, 239, 146],
            #               'Spermatogonia': [249, 123, 215]}
            # for gene, tmp_color in cell_color.items():
            #     if gene in legend:
            #         colors[legend[gene]] = tmp_color

        except:
            print("Error color import ,use random color!")
            pass

        if special_cluster_dict != {}:  # 指定展示模式
            for special_cluster, special_color in special_cluster_dict.items():
                if special_color is not None:
                    colors[legend[special_cluster]] = special_color
        np.save(os.path.join(save_dir, "clusters_colors.npy"), colors)

    #  画一个图例
    legend_color_dict = {cluster_name: colors[tmp_cluster_num] for cluster_name, tmp_cluster_num in legend.items()}
    legend_pic = draw_legend(legend_color_dict)
    tifffile.imwrite(os.path.join(save_dir, "legend.tif"), legend_pic)
    # exit(0)

    #  根据类型颜色对应细胞颜色库
    cell_color = np.zeros((cells.max() + 1, 3), np.uint8)
    for i in range(len(cell_color)):
        if i in cluster_dict:
            cell_color[i] = colors[cluster_dict[i]]
        # elif i>0:
        #     cell_color[i] = cell_color[i - 1]

    np.save(os.path.join(save_dir, "colors.npy"), cell_color)

    # 画可视化图像
    cell_color_img = cell_color[cells]
    tifffile.imwrite(os.path.join(save_dir, "cell_cluster_color_img.tif"), cell_color_img, compression="jpeg")

    # 画细胞边界
    outline = utils.masks_to_outlines(cells)
    cell_color_img[outline] = 255 - cell_color_img[outline]
    tifffile.imwrite(os.path.join(save_dir, "cell_cluster_color_outline_img.tif"), cell_color_img, compression="jpeg")

    # 加图例
    cell_color_img = add_legend(cell_color_img, legend_pic)
    tifffile.imwrite(os.path.join(save_dir, "cell_cluster_with_legend_img.tif"), cell_color_img, compression="jpeg")
    # tifffile.imwrite(os.path.join(save_dir, "cell_cluster_with_legend_img{}.tif".format(sub_index)), cell_color_img, compression="jpeg")

