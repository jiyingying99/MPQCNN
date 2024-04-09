import matplotlib.pyplot as plt
import numpy as np

# 模拟数据
environments = ['-4', '-2', '0', '2', '4', '6']
#对比实验
# grades_A = [76.33, 83.60, 89.67, 91.73, 91.93, 94.27]  # DCA-BiGRU
# grades_B = [69.73, 82.93, 90.87, 95.80, 95.73, 97.27]  # wdcnn
# grades_C = [74.73, 84.53, 91.80, 96.60, 99.47, 99.80]  # qcnn
# grades_D = [79.87, 89.33, 97.33, 97.13, 97.13, 98.73]  # ews
# grades_E = [81.87, 93.27, 96.80, 98.80, 99.47, 99.87]  # my
# 消融实验
grades_A = [67.73, 78.20, 87.67, 94.20, 96.80, 98.07]  # A网络
grades_B = [81.67, 91.80, 95.87, 98.33, 99.47, 99.93]  # B网络
grades_C = [79.53, 89.60, 95.60, 97.87, 99.13, 99.73]  # C网络
grades_D = [81.87, 93.27, 96.80, 98.80, 99.47, 99.87]  # D网络
#jnu无噪
# grades_A = [67.73, 78.20, 87.67, 94.20, 96.80, 98.07]  # A网络
# grades_B = [81.67, 91.80, 95.87, 98.33, 99.47, 99.93]  # B网络
# grades_C = [79.53, 89.60, 95.60, 97.87, 99.13, 99.73]  # C网络
# grades_D = [81.87, 93.27, 96.80, 98.80, 99.47, 99.87]  # D网络

# 设置柱状图和折线图的位置
x = np.arange(len(environments))
width = 0.15

# 画图
fig, ax = plt.subplots()
# bars_A = ax.bar(x - 2*width, grades_A, width, label='A')
# bars_B = ax.bar(x - width, grades_B, width, label='B')
# bars_C = ax.bar(x, grades_C, width, label='C')
# bars_D = ax.bar(x + width, grades_D, width, label='D')
# bars_E = ax.bar(x + width, grades_D, width, label='E')

bars_A = ax.bar(x - 2*width, grades_A, width, color='red', label='A')
bars_B = ax.bar(x - width, grades_B, width, color='g', label='B')
bars_C = ax.bar(x, grades_C, width, color='blue', label='C')
bars_D = ax.bar(x + width, grades_D, width, color='orange', label='MPQCNN')
# bars_E = ax.bar(x + 2*width, grades_E, width, color='purple', label='MPQCNN')

# fig, ax = plt.subplots()
# bars_A = ax.bar(x - 2*width, grades_A, width, color='r', label='A', hatch='.')
# bars_B = ax.bar(x - width, grades_B, width, color='g', label='B', hatch='x')
# bars_C = ax.bar(x, grades_C, width, color='b', label='C', hatch='/')
# bars_D = ax.bar(x + width, grades_D, width, color='y', label='D', hatch='-')

# 添加标签、标题和图例
ax.set_xlabel('SNR(db)')
ax.set_ylabel('Accuracy(%)')
# ax.set_title('Grades in Different Environments')
ax.set_xticks(x)
ax.set_xticklabels(environments)
plt.subplots_adjust(right=0.75)
# 设置 y 轴范围从 40 开始
ax.set_ylim(65, None)
ax.legend(loc='upper left', bbox_to_anchor=(0.5, 1.15))


# plt.savefig('../pic/cwru_xiaorong.jpg', dpi=500)
# 显示图形
plt.show()





#
# import matplotlib.pyplot as plt
#
# # 假设有四种环境和四个不同成绩
# environments = ['Env1', 'Env2', 'Env3', 'Env4']
# scores_A = [85, 90, 88, 92]  # 成绩A
# scores_B = [78, 85, 80, 88]  # 成绩B
# scores_C = [90, 92, 87, 94]  # 成绩C
# scores_D = [82, 88, 85, 90]  # 成绩D
#
# # 创建柱状图
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
#
# for i, ax in enumerate(axes.flat):
#     if i == 0:
#         ax.bar(environments, scores_A, color='blue', hatch='/', label='Score A')
#     elif i == 1:
#         ax.bar(environments, scores_B, color='green', hatch='\\', label='Score B')
#     elif i == 2:
#         ax.bar(environments, scores_C, color='red', hatch='x', label='Score C')
#     elif i == 3:
#         ax.bar(environments, scores_D, color='orange', hatch='.', label='Score D')
#
#     ax.set_title(f'Environment {i+1}')
#     ax.set_ylabel('Score')
#     ax.set_xlabel('Environment')
#     ax.legend()
#
# plt.tight_layout()
# plt.show()
#
# # 创建折线图
# plt.figure(figsize=(8, 6))
#
# plt.plot(environments, scores_A, marker='o', label='Score A', color='blue')
# plt.plot(environments, scores_B, marker='s', label='Score B', color='green')
# plt.plot(environments, scores_C, marker='^', label='Score C', color='red')
# plt.plot(environments, scores_D, marker='d', label='Score D', color='orange')
#
# plt.title('Scores in Different Environments')
# plt.xlabel('Environment')
# plt.ylabel('Score')
# plt.legend()
# plt.grid(True)
#
# plt.show()



