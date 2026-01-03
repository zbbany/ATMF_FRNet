#"HKU-IS"
#y = [10.1, 10.0, 8.1, 9.2, 8.7, 9.2, 9.2, 9.2, 7.2, 8.4, 7]
#y_values = [7, 8.4, 9.3, 9.4, 7.2, 9.2, 9.2, 9.2, 8.7, 9.2, 8.1, 10.0, 10.1]

#"ECSSD"
#y = [9.2, 8.7, 7.9, 8.3, 7.3, 7.6, 8.1, 8.4, 6.3, 7.4, 6.2]
#y_values = [6.2, 7.4, 8.1, 7.1, 6.3, 8.4, 8.1, 7.6, 7.3, 8.3, 7.9, 8.7, 9.2]

#"DUTS"
#y = [13.7, 13.8, 10.7, 15.1, 12.3, 12.6, 13.1, 13.0, 10.6, 11.8, 9.4]
#y_values = [9.4, 11.8, 13.9, 14.9, 10.6, 13.0, 13.1, 12.6, 12.3, 11.9, 10.7, 13.8, 13.7]

#"OMRON"
#y = [17.5, 19.9, 17.2, 14.9, 17.8, 18.8, 18.4, 18.6, 14.4, 16.6, 13]
#y_values = [13, 16.6, 21.9, 17.5, 14.4, 18.6, 18.4, 18.8, 17.8, 14.9, 17.2, 19.9, 17.5]

#"PASCAL-S"
#y_values = [10, 13.3, 11.9, 10.5, 13, 12, 16.2, 12.9, 12.2, 12, 14, 15.1]


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 全局设置字体为 Times New Roman（保持原有字号）
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.sans-serif'] = ['Times New Roman']  # 兼容性设置

# 数据准备
models = ['Ours','DIPONet', 'MGuidNet', 'ICON', 'CII', 'DFI',
         'GCPANet', 'MINet', 'ITSD', 'F3Net', 'EGNet', 'BASNet']
y_values = [10, 13.3, 11.9, 10.5, 13, 12, 16.2, 12.9, 12.2, 12, 14, 15.1]
positions = list(range(1, len(models) + 1))

colors = [
        'red',          # Ours (红色，突出显示)
        '#228B22',      # ITSD (深绿)
        '#FFD700',      # DFI (金黄)
        '#696969',      # MGuidNet (深灰)
        '#00FFFF',      # EGNet (青蓝)
        '#4B0082',      # MINet (靛蓝，替代重复的深蓝)
        '#32CD32',      # CII (酸橙绿，替代重复的深绿)
        '#9370DB',      # DIPONet (中紫)
        '#FFA500',      # F3Net (亮橙)
        '#FF00FF',      # GCPANet (品红)
        '#8FBC8F',      # ICON (浅灰绿，替代重复的金黄)
        '#8A2BE2'       # DSLRDNet (蓝紫色，替代中紫)
    ]

plt.figure(figsize=(8, 9))  # 宽度为8，高度为12

# 创建水平条形图
bars = plt.barh(y=positions, width=y_values, alpha=0.8, color=colors, height=0.8)

# 设置y轴标签（模型名称）
plt.yticks(positions, models, fontsize=30, va='center')  # 字号保持30
plt.tick_params(axis='y', which='both', left=False)  # 隐藏y轴刻度线

# 设置坐标轴样式
ax = plt.gca()
# 设置实线边框
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.5)
    spine.set_color('black')

# 添加虚线网格线
ax.grid(axis='x', linestyle='--', alpha=0.5, color='gray')

# 添加数值标签（补充字体设置）
for bar, val in zip(bars, y_values):
    ax.text(bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}',
            va='center', ha='left',
            fontsize=40, color='black',
            fontname='Times New Roman')  # 显式设置字体

# 调整坐标轴范围
plt.xlim(9.8, 17.8)
plt.ylim(0.4, len(models) + 0.6)

# 隐藏x轴
plt.xticks([])
ax.tick_params(axis='x', length=0)

# 调整边距
plt.subplots_adjust(left=0.257, right=0.967, top=0.99, bottom=0.021)

plt.show()