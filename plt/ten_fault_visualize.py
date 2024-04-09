'''
绘图程序
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio


def draw_data(matfilepath, x, y, z, xlabel):
    raw_data = scio.loadmat(matfilepath)
    # 读取内容
    signal = ''
    for key, value in raw_data.items():
        # if key[5:7] == 'DE':
        if key == 'reshaped_data':
            signal = value

    time = [i for i in range(2048)]
    axis = np.random.randint(2048)

    # Plot colors numbers
    ax = plt.subplot(x, y, z)
    ax.plot(time, signal[axis:axis + 2048], color='mediumturquoise')

    plt.ylabel('A(mm)', fontdict={'family': 'Times New Roman', 'size': 18}, )
    plt.xlabel(xlabel,fontdict={'family': 'Times New Roman', 'size': 18}, )
    plt.yticks(fontproperties='Times New Roman', size=18)
    plt.xticks(fontproperties='Times New Roman', size=18)


# 设置xtick和ytick的方向：in、out、inout
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.figure(figsize=(12, 12))  # 宽12，高8
draw_data('../plt/gradcam/reshaped_data_datanew_0.mat', x=5, y=2, z=1, xlabel='(a)B007')
# draw_data('../data/0HP/12k_Drive_End_B007_0_118.mat', x=5, y=2, z=1, xlabel='(a)B007')
draw_data('../data/0HP/12k_Drive_End_B014_0_185.mat', x=5, y=2, z=2, xlabel='(b)B014')
draw_data('../data/0HP/12k_Drive_End_B021_0_222.mat', x=5, y=2, z=3, xlabel='(c)B021')
draw_data('../data/0HP/12k_Drive_End_IR007_0_105.mat', x=5, y=2, z=4, xlabel='(d)IR007')
draw_data('../data/0HP/12k_Drive_End_IR014_0_169.mat', x=5, y=2, z=5, xlabel='(e)IR014')
draw_data('../data/0HP/12k_Drive_End_IR021_0_209.mat', x=5, y=2, z=6, xlabel='(f)IR021')
draw_data('../data/0HP/12k_Drive_End_OR007@6_0_130.mat', x=5, y=2, z=7, xlabel='(g)OR007')
draw_data('../data/0HP/12k_Drive_End_OR014@6_0_197.mat', x=5, y=2, z=8, xlabel='(h)OR014')
draw_data('../data/0HP/12k_Drive_End_OR021@6_0_234.mat', x=5, y=2, z=9, xlabel='(i)OR021')
draw_data('../data/0HP/normal_0_97.mat', x=5, y=2, z=10, xlabel='(j)normal')

# draw_data('../data/0HP_TrainNoised/12k_Drive_End_B007_0_118.mat', x=5, y=2, z=1, xlabel='(a)B007')
# draw_data('../data/0HP_TrainNoised/12k_Drive_End_B014_0_185.mat', x=5, y=2, z=2, xlabel='(a)B007')
# draw_data('../data/0HP_TrainNoised/12k_Drive_End_B021_0_222.mat', x=5, y=2, z=3, xlabel='(a)B007')
# draw_data('../data/0HP_TrainNoised/12k_Drive_End_IR007_0_105.mat', x=5, y=2, z=4, xlabel='(a)B007')
# draw_data('../data/0HP_TrainNoised/12k_Drive_End_IR014_0_169.mat', x=5, y=2, z=5, xlabel='(a)B007')
# draw_data('../data/0HP_TrainNoised/12k_Drive_End_IR021_0_209.mat', x=5, y=2, z=6, xlabel='(a)B007')
# draw_data('../data/0HP_TrainNoised/12k_Drive_End_OR007@6_0_130.mat', x=5, y=2, z=7, xlabel='(a)B007')
# draw_data('../data/0HP_TrainNoised/12k_Drive_End_OR014@6_0_197.mat', x=5, y=2, z=8, xlabel='(a)B007')
# draw_data('../data/0HP_TrainNoised/12k_Drive_End_OR021@6_0_234.mat', x=5, y=2, z=9, xlabel='(a)B007')
# draw_data('../data/0HP_TrainNoised/normal_0_97.mat', x=5, y=2, z=10, xlabel='(a)B007')

plt.tight_layout()  # 自动调整各子图间距
# plt.savefig('../pic/0hp_ten_faults_view_-6.svg', dpi=600)
plt.show()
