import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import os
import textwrap
from matplotlib.patches import Rectangle, ConnectionPatch  # 导入Rectangle和ConnectionPatch


def plot_pr_curves(models, dataset='OMRON', output_filename='pr_curves.svg'):

    os.makedirs('Fm_curves', exist_ok=True)  # 确保保存目录存在
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 40  # 设置全局字体大小为24号
    plt.figure(figsize=(8, 12))  # 宽度为8，高度为12
    plt.subplots_adjust(left=0.296, right=0.745, top=0.98, bottom=0.115)

    # 定义颜色列表（可自定义）
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
    for model, color in zip(models, colors):
        try:
            # 加载PR数据（假设已存在）
            rec_path = f'pr_curves/{model}_{dataset}_rec.mat'
            prec_path = f'pr_curves/{model}_{dataset}_prec.mat'

            if not os.path.exists(rec_path) or not os.path.exists(prec_path):
                print(f"Warning: {model} PR data not found. Skipping F-measure calculation.")
                continue

            rec_data = sio.loadmat(rec_path)
            prec_data = sio.loadmat(prec_path)
            recalls = rec_data['recalls'].flatten()
            precisions = prec_data['precisions'].flatten()

            # 生成阈值数组（假设阈值从高到低排列，这里用均匀分布示例）
            num_points = len(recalls)
            thresholds = np.linspace(255, 0, num_points)  # 根据实际情况调整阈值范围

            # 计算F-measure
            fmeasures = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

            # 保存结果
            sio.savemat(
                f'Fm_curves/{model}_{dataset}_Thre.mat',
                {'thresholds': thresholds}
            )
            sio.savemat(
                f'Fm_curves/{model}_{dataset}_Fmeasure.mat',
                {'fmeasures': fmeasures}
            )

            # 绘制曲线（修改部分）
            linestyle = '-' if model == 'Ours' else '--'  # 设置线型
            plt.plot(thresholds, fmeasures,
                     label=model,
                     color=color,
                     linewidth=3,
                     linestyle=linestyle)  # 添加线型参数

        except Exception as e:
            print(f"Error processing {model}: {str(e)}")

    # 设置坐标轴标签（保持不变）
    plt.xlabel('Threshold', fontsize=40, fontfamily='Times New Roman')
    plt.ylabel('F-measure', fontsize=40, fontfamily='Times New Roman')

    # 设置图例（保持不变）
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncol=3, fontsize=24,
               frameon=True, facecolor='white', edgecolor='none',
               prop={'family': 'Times New Roman', 'size': 30},
               handletextpad=0.05,
               columnspacing=0.3,
               labelspacing=0.05
               )
    plt.grid(alpha=0.3)
    plt.xlim(0, 255)
    plt.ylim(0, 1)

    plt.gca().tick_params(axis='both', which='major', width=3)

    # 添加放大区域
    ax = plt.gca()
    ax_zoom = plt.axes([0.32, 0.4, 0.35, 0.3])  # 定义放大图的位置和大小

    for model, color in zip(models, colors):
        try:
            rec_path = f'pr_curves/{model}_{dataset}_rec.mat'
            prec_path = f'pr_curves/{model}_{dataset}_prec.mat'

            if not os.path.exists(rec_path) or not os.path.exists(prec_path):
                continue

            rec_data = sio.loadmat(rec_path)
            prec_data = sio.loadmat(prec_path)
            recalls = rec_data['recalls'].flatten()
            precisions = prec_data['precisions'].flatten()

            num_points = len(recalls)
            thresholds = np.linspace(255, 0, num_points)
            fmeasures = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

            # 筛选出放大区域的数据
            mask = (thresholds >= 50) & (thresholds <= 200) & (fmeasures >= 0.5) & (fmeasures <= 1)
            linewidth = 5 if model == 'Ours' else 3  # 修改点：线宽差异化
            linestyle = '-' if model == 'Ours' else '--'

            ax_zoom.plot(thresholds[mask], fmeasures[mask],
                         label=model,
                         color=color,
                         linewidth=linewidth,  # 使用动态线宽
                         linestyle=linestyle)

        except Exception as e:
            print(f"Error processing {model} for zoom: {str(e)}")

    # 在主图上绘制一个矩形框，指示放大区域左上角
    rect = Rectangle((50, 0.77), 150, 0.07, linewidth=2, edgecolor='black', facecolor='none', linestyle='-', zorder=100)  #########
    ax.add_patch(rect)

    # 在放大图上绘制一个矩形框，指示放大区域的位置
    rect_zoom = Rectangle((50, 0.5), 150, -0.08, linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
    ax_zoom.add_patch(rect_zoom)

    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])

    # 设置放大图的坐标轴范围和标签
    ax_zoom.set_xlim(50, 200)
    ax_zoom.set_ylim(0.77, 0.84)  ################
    ax_zoom.grid(alpha=0.3)


    # 绘制连接线
    # 获取主图和放大图的坐标轴范围
    zoom_xlim = ax_zoom.get_xlim()
    zoom_ylim = ax_zoom.get_ylim()

    rect_left = 50
    rect_bottom = 0.77
    rect_width = 150

    # 计算连接线的坐标
    # 主图左下角到放大图左上角
    con1_xyA = (rect_left, rect_bottom)  # 定位框左下角
    con1_xyB = (zoom_xlim[0], zoom_ylim[1])  # 放大图左上角
    con1 = ConnectionPatch(xyA=con1_xyA, xyB=con1_xyB, coordsA="data", coordsB="data", axesA=ax, axesB=ax_zoom,
                           color="black", linestyle="--")
    ax.add_patch(con1)
    for spine in ax.spines.values():
        spine.set_linewidth(3)  # 设置边框线宽度
    ax.spines['left'].set_position('zero')  # 将y轴移到0位置
    ax.spines['bottom'].set_position('zero')  # 将x轴移到0位置


    # 主图右下角到放大图右上角
    con2_xyA = (rect_left + rect_width, rect_bottom)  # 定位框右下角
    con2_xyB = (zoom_xlim[1], zoom_ylim[1])  # 放大图右上角
    con2 = ConnectionPatch(xyA=con2_xyA, xyB=con2_xyB, coordsA="data", coordsB="data", axesA=ax, axesB=ax_zoom,
                           color="black", linestyle="--")
    ax.add_patch(con2)

    plt.savefig(output_filename, format='svg', bbox_inches='tight', pad_inches=0.05)
    plt.show()


if __name__ == "__main__":
    models = [
        'BASNet', 'ITSD', 'DFI', 'MGuidNet', 'Ours', 'EGNet',
        'MINet', 'CII', 'DIPONet', 'F3Net', 'GCPANet', 'ICON',
        'DSLRDNet'
    ]

    plot_pr_curves(models, dataset='OMRON', output_filename='custom_pr_curves.svg')


