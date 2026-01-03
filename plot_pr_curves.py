import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle, ConnectionPatch  # 导入Rectangle和ConnectionPatch

def plot_pr_curves(models, dataset='ECSSD', output_filename='pr_curves.svg'):
    # 设置全局字体和字号
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 40  # 设置全局字体大小为24号

    # 调整图像大小为竖向长方形
    plt.figure(figsize=(8, 16))  # 宽度为8，高度为12
    plt.subplots_adjust(left=0.4, right=0.6, top=0.98, bottom=0.115)

    colors = [
        '#2F4F8F',  # BASNet (深蓝)
        '#228B22',  # ITSD (深绿)
        '#FFD700',  # DFI (金黄)
        '#696969',  # MGuidNet (深灰)
        'red',  # Ours (红色，突出显示)
        '#00FFFF',  # EGNet (青蓝)
        '#4B0082',  # MINet (靛蓝，替代重复的深蓝)
        '#32CD32',  # CII (酸橙绿，替代重复的深绿)
        '#9370DB',  # DIPONet (中紫)
        '#FFA500',  # F3Net (亮橙)
        '#FF00FF',  # GCPANet (品红)
        '#8FBC8F',  # ICON (浅灰绿，替代重复的金黄)
        '#8A2BE2'  # DSLRDNet (蓝紫色，替代中紫)
    ]
    ax = plt.gca()  # 获取主坐标轴对象

    for model, color in zip(models, colors):
        try:
            rec_data = sio.loadmat(f'pr_curves/{model}_{dataset}_rec.mat')
            prec_data = sio.loadmat(f'pr_curves/{model}_{dataset}_prec.mat')

            recalls = rec_data['recalls'].flatten()
            precisions = prec_data['precisions'].flatten()

            sort_idx = np.argsort(recalls)[::-1]

            # 设置线宽和线型
            linewidth = 5 if model == 'Ours' else 3  # 修改点：线宽差异化
            linestyle = '-' if model == 'Ours' else '--'

            ax.plot(recalls[sort_idx], precisions[sort_idx],
                    label=model,
                    color=color,
                    linewidth=linewidth,  # 使用动态线宽
                    linestyle=linestyle)
        except FileNotFoundError:
            print(f"Warning: {model} data not found")

    # 创建嵌入图（放大区域）
    axins = inset_axes(ax,
                      width=3.7,     # 放大宽度倍数
                      height=3.7,  # 放大高度倍数
                      loc='upper right',
                      bbox_to_anchor=(0.68, 0.83),  # 锚点位置（相对于主图坐标系）
                      bbox_transform=ax.transAxes)

    # 设置嵌入图坐标范围
    axins.set_xlim(0.88, 0.98)
    axins.set_ylim(0.88, 0.98)

    # 隐藏嵌入图的刻度
    axins.xaxis.set_visible(False)
    axins.yaxis.set_visible(False)

    # 在嵌入图中绘制所有曲线（继承主图样式）
    for model, color in zip(models, colors):
        try:
            rec_data = sio.loadmat(f'pr_curves/{model}_{dataset}_rec.mat')
            prec_data = sio.loadmat(f'pr_curves/{model}_{dataset}_prec.mat')

            recalls = rec_data['recalls'].flatten()
            precisions = prec_data['precisions'].flatten()

            sort_idx = np.argsort(recalls)[::-1]

            # 筛选0.8-1范围内的数据点
            mask = (recalls >= 0.6) & (recalls <= 1.0)
            axins.plot(recalls[sort_idx][mask], precisions[sort_idx][mask],
                      color=color,
                      linewidth=3,
                      linestyle='-' if model == 'Ours' else '--')
        except FileNotFoundError:
            continue

    # 在主图上绘制一个矩形框，指示放大区域
    rect_left = 0.88
    rect_bottom = 0.88
    rect_width = 0.1
    rect_height = 0.1
    rect = Rectangle((rect_left, rect_bottom), rect_width, rect_height, linewidth=2, edgecolor='black', facecolor='none', linestyle='-', zorder=100)
    ax.add_patch(rect)

    # 获取嵌入图的坐标轴范围
    zoom_xlim = axins.get_xlim()
    zoom_ylim = axins.get_ylim()

    # 绘制连接线
    # 主图左上角到嵌入图左上角
    con1_xyA = (rect_left, rect_bottom + rect_height)  # 主图定位框左上角
    con1_xyB = (zoom_xlim[0], zoom_ylim[1])  # 嵌入图左上角
    con1 = ConnectionPatch(xyA=con1_xyA, xyB=con1_xyB, coordsA="data", coordsB="data", axesA=ax, axesB=axins, color="black", linestyle="--")
    ax.add_patch(con1)

    # 主图右下角到嵌入图右下角
    con2_xyA = (rect_left + rect_width, rect_bottom)  # 主图定位框右下角
    con2_xyB = (zoom_xlim[1], zoom_ylim[0])  # 嵌入图右下角
    con2 = ConnectionPatch(xyA=con2_xyA, xyB=con2_xyB, coordsA="data", coordsB="data", axesA=ax, axesB=axins, color="black", linestyle="--")
    ax.add_patch(con2)

    # 设置x轴和y轴从0到1的大框线加粗
    for spine in ax.spines.values():
        spine.set_linewidth(3)  # 设置边框线宽度
    ax.spines['left'].set_position('zero')  # 将y轴移到0位置
    ax.spines['bottom'].set_position('zero')  # 将x轴移到0位置


    # 主图设置
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Recall', fontsize=40, fontname='Times New Roman')
    ax.set_ylabel('Precision', fontsize=40, fontname='Times New Roman')
    ax.grid(alpha=0.3)

    # 图例设置
    ax.legend(loc='lower center', fontsize=24, frameon=True, edgecolor='none',
              prop={'family': 'Times New Roman', 'size': 30}, ncol=2,
              handletextpad=0.05,  # 调整句柄与文本之间的填充
              columnspacing=0.3,  # 调整列之间的间距
              labelspacing=0.05)  # 调整标签之间的垂直间距

    # 保存图像，只保留有内容的部分
    plt.savefig(output_filename, format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.show()

if __name__ == "__main__":


    models = [
        'BASNet', 'ITSD', 'DFI', 'MGuidNet', 'Ours', 'EGNet',
        'MINet', 'CII', 'DIPONet', 'F3Net', 'GCPANet', 'ICON',
        'DSLRDNet'
    ]
    plot_pr_curves(models, output_filename='custom_pr_curves.svg')

