import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# 准备数据
index = [1,2,3,4,5]

values_hit_a1 = [3.319551467895508,	4.399291515350342,	7.681039810180664,	8.009450912,	7.269935131072998 ]

values_hit_a5 = [8.29887867,	10.740697860717773,	18.346132278442383,	19.196693420410156,	16.940343856811523]

values_hit_c1 = [7.577082633972168,	8.009450912,	7.428429,	4,	2.391021966934204]

values_hit_c5 = [17.909038543701172,	19.196693420410156,	16.92459,	9.637331008911133,	6.485528945922852]

fig, axs = plt.subplots(1, 2, figsize=(12,5))

#2D59C6 #318EDE
axs[0].plot(index,values_hit_a5, label='Hit@5',marker ='s',color='#26CDD5',linewidth=4,markersize='14')  # 添加图例

axs[0].plot(index,values_hit_a1, label='Hit@1',marker = '^',color='#2D59C6',linewidth=4, markersize='14')  # 添加图例
axs[0].set_title('')  # 设置子图标题
axs[0].set_xlabel('')  # 设置X轴标签
axs[0].set_ylabel('')  # 设置Y轴标签

axs[0].tick_params(axis='y', labelsize=15)
axs[0].set_ylabel('Hit Rate(%)', fontsize=18, fontweight='semibold')
axs[0].set_xlabel('Layer Number L1', fontsize=18,fontweight='semibold')
axs[0].set_xlim(0.5, 5.5)
axs[0].set_ylim(2, 20)
axs[0].grid(True, linestyle='--', alpha=0.7)
axs[0].legend(labelspacing=1.5,fontsize=14)  # 显示图例

for i in range(len(index)):
    axs[0].text(i+1, values_hit_a1[i]+0.5, "{0:.3f}".format(values_hit_a1[i]), ha='center', va='bottom',fontsize=14)

for i in range(len(index)):
    axs[0].text(i+1, values_hit_a5[i]-1.5, "{0:.3f}".format(values_hit_a5[i]), ha='center', va='bottom',fontsize=14)

axs[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
axs[0].xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))



axs[1].plot(index,values_hit_c5, label='Hit@5',marker ='s',color='#26CDD5',linewidth=4, markersize='14')  # 添加图例
axs[1].plot(index,values_hit_c1, label='Hit@1',marker = '^',color='#2D59C6',linewidth=4, markersize='14')  # 添加图例
axs[1].set_title('')  # 设置子图标题
axs[1].set_xlabel('Layer Number L2',fontsize=18, fontweight='semibold')  # 设置X轴标签
axs[1].set_ylabel('')  # 设置Y轴标签



axs[1].tick_params(axis='y', labelsize=15)
axs[1].set_ylabel('Hit Rate(%)', fontsize=15, fontweight='semibold')
axs[1].set_ylim(2, 20)
axs[1].set_xlim(0.5, 5.5)
axs[1].grid(True, linestyle='--', alpha=0.7)

for i in range(len(index)):
    axs[1].text(i+1, values_hit_c1[i]+0.5, "{0:.3f}".format(values_hit_c1[i]), ha='center', va='bottom',fontsize=14)

for i in range(len(index)):
    axs[1].text(i+1, values_hit_c5[i]-1.5, "{0:.3f}".format(values_hit_c5[i]), ha='center', va='bottom',fontsize=14)

axs[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
axs[1].xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
axs[1].legend(labelspacing=1.,fontsize=14)  # 显示图例

plt.tight_layout()


# 显示图形
plt.show()

