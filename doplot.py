import numpy as np
import matplotlib.pyplot as plt

# # doplot.
# plt.title("Training time cost per epoch (MLP)")
# xs_sync = [1, 2, 3, 4]
# xs_async = [5, 6, 7, 8]
# y_sync = [7.29, 4.80, 37.28, 45.63] # [76.28, 40.69, 146.85, 116.68]# [13.82, 7.93, 105.38, 90.18] # [7.29, 4.80, 37.28, 45.63]
# y_async = [6.98, 3.47, 7.38, 3.22] # [75.95, 37.78, 72.10, 31.83] # [13.54, 6.83, 16.96, 6.62] # [6.98, 3.47, 7.38, 3.22]
# ticks_sync = ['1s1g', '1s2g', '2s2g', '2s4g']
# ticks_async = ['1s1g', '1s2g', '2s2g', '2s4g']
# plt.bar(xs_sync, y_sync, label='sync')
# plt.bar(xs_async, y_async, label='async')
# plt.xticks(xs_sync+xs_async, ticks_sync+ticks_async)
# plt.xlabel('Different experimental settings')
# plt.ylabel('Time cost (s)')
# plt.legend(loc='best')
# plt.savefig('MLP.png', bbox_inches='tight')
# plt.show()

# network
ins, outs = [], []
with open("sync_net.txt", "r") as f:
    for line  in f:
        _line = line.strip().split()
        if len(_line) <= 0:
            break
        ins.append(float(_line[0]))
        outs.append(float(_line[1]))
ins = [x/150 for x in ins[10:-80]]
outs = [x/150 for x in outs[10:-80]]
xs = range(len(ins))
plt.title('Network traffic of sync mode')
plt.xlabel('time (s)')
plt.ylabel('rate (mb/s)')
plt.plot(xs, ins, label='in')
plt.plot(xs, outs, label='out')
plt.legend(loc='best', fontsize = 'x-large')
plt.savefig('Sync_Net.png', bbox_inches='tight')
plt.show()

# # doplot.
# xs_sync = [1, 2, 3, 4]
# xs_async = [5, 6, 7, 8]
# mlp_y_sync = [7.29, 4.80, 37.28, 45.63]
# cnn_y_sync = [13.82, 7.93, 105.38, 90.18]
# resnet_y_sync = [76.28, 40.69, 146.85, 116.68]
# mlp_y_async = [6.98, 3.47, 7.38, 3.22]
# cnn_y_async = [13.54, 6.83, 16.96, 6.62]
# resnet_y_async = [75.95, 37.78, 72.10, 31.83]
# columns = ('1s1g', '1s2g', '2s2g', '2s4g')
# rows = ['MLP', 'CNN', 'RESNET']
# mlp_y = [mlp_y_sync[i]/mlp_y_async[i] - cnn_y_sync[i]/cnn_y_async[i] for i in range(4)]
# cnn_y = [cnn_y_sync[i]/cnn_y_async[i] - resnet_y_sync[i]/resnet_y_async[i] for i in range(4)]
# resnet_y = [resnet_y_sync[i]/resnet_y_async[i] for i in range(4)]
# data = [resnet_y,
#         cnn_y,
#         mlp_y]
# colors = plt.cm.BuPu(np.linspace(0.2, 0.8, len(rows)))
# index = np.arange(len(columns)) + 0.3
# bar_width = 0.4
# # Initialize the vertical-offset for the stacked bar chart.
# y_offset = np.zeros(len(columns))
# cell_text = []
# # Plot bars and create text labels for the table
# for row in range(len(rows)):
#     # print("row", row)
#     plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
#     y_offset = y_offset + data[row]
#     cell_text.append(['%1.1f' % x for x in y_offset])
# # Reverse colors and text labels to display the last value at the top.
# colors = colors[::-1]
# cell_text.reverse()
# # Add a table at the bottom of the axes
# the_table = plt.table(cellText=cell_text,
#                       rowLabels=rows,
#                       rowColours=colors,
#                       colLabels=columns,
#                       loc='bottom')
# # Adjust layout to make room for the table:
# plt.subplots_adjust(left=0.2, bottom=0.2)
# plt.ylabel("Ratio of time cost")
# #plt.yticks(values * value_increment, ['%d' % val for val in values])
# plt.xticks([])
# plt.title("Comparison of async and sync mode under multi-device settings (TF)")
# plt.savefig('Sync_and_Async_modes.png', bbox_inches='tight')
# plt.show()

#
# # doplot.
# mlp_s = [[998, 997, 2026, 1999], [106, 107, 1864, 1836]]
# cnn_s = [[151, 153, 242, 242], [28, 20, 764, 748]]
# resnet_s = [[181, 183, 221, 203], [141, 140, 656, 631]]
# y_s = [mlp_s, cnn_s, resnet_s]
# for i in range(3):
#     for j in range(2):
#         total = 0.0
#         for k in range(len(y_s[i][j])):
#             total += y_s[i][j][k]
#         for k in range(len(y_s[i][j])):
#             y_s[i][j][k] /= total
# y_s = [y[0] + y[1] for y in y_s]
# x_s = [4, 8, 12]
# label_s = ['sync_worker3', 'sync_worker2', 'sync_worker1', 'sync_worker0', 'async_worker3', 'async_worker2', 'async_worker1', 'async_worker0']
# dist_s = [-1.1, -0.8, -0.5, -0.2, 0.2, 0.5, 0.8, 1.1]
# colors = [(0, 0.4, 0.4), (0, 0.3, 0.3), (0, 0.8, 0.8), (0, 0.7, 0.7), (0.4, 0.4, 0), (0.3, 0.3, 0), (0.8, 0.8, 0), (0.7, 0.7, 0)]
# for j in range(len(y_s[0])):
#     plt.bar([x+dist_s[j] for x in x_s], [y_s[k][j] for k in range(3)], alpha=0.9, width=0.28, color=colors[j], label=label_s[j])
# plt.title("Steps taken by different workers (num=4)")
# plt.xticks(x_s, ['MLP', 'CNN', 'RESNET'])
# plt.xlabel('Different experimental settings')
# plt.ylabel('conducted steps (percent, %)')
# plt.legend(loc='best', fontsize='x-small')
# plt.savefig('steps_worker4.png', bbox_inches='tight')
# plt.show()
