import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# 准备数据
categories = ['POI-Enhancer/P', 'POI-Enhancer/D','POI-Enhancer/F', 'POI-Enhancer/C', 'POI-Enhancer']
values1 = [7.22977, 7.036031, 7.42843, 6.712345, 8.00945091247558]
values2 = [0.506885, 0.510757,0.509446, 0.475903614457831, 0.518932874354561]
values3 = [0.3683925, 0.34671673,0.378554, 0.38322908, 0.3133186]
index = range(len(categories))

colors= ['#76E0D6','#26CDD5', '#318EDE', '#2D59C6','#1F4E79']
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].bar(categories, values1, color=colors, label=categories)
axs[0].set_title('POI Recommendation',fontsize=14,fontweight='bold')
axs[0].set_xticklabels([])
axs[0].tick_params(axis='y', labelsize=13)
axs[0].set_ylabel('Hit@1(%)', fontsize=15, fontweight='semibold')
axs[0].set_ylim(6.2, 8.3)

for i in index:
    axs[0].text(i, values1[i], "{0:.3f}".format(values1[i]), ha='center', va='bottom',fontsize=12, fontweight='semibold')


axs[1].bar(categories, values2, color=colors)
axs[1].set_title('Check-in Sequence Classification',fontsize=14,fontweight='bold')
axs[1].set_xticklabels([])
axs[1].tick_params(axis='y', labelsize=13)
axs[1].set_ylabel('Acc',fontsize=15, fontweight='semibold')
axs[1].set_ylim(0.45, 0.55)

for i in index:
    axs[1].text(i, values2[i],  "{0:.3f}".format(values2[i]),ha='center', va='bottom',fontsize=12, fontweight='semibold')

axs[2].bar(categories, values3, color=colors)
axs[2].set_title('POI Visitors Flow Prediction',fontsize=14,fontweight='bold')
axs[2].set_xticklabels([])
axs[2].tick_params(axis='y', labelsize=13)
axs[2].set_ylabel('MAE', fontsize=15, fontweight='semibold')
axs[2].set_ylim(0.25, 0.4)

for i in index:
    axs[2].text(i, values3[i] , "{0:.3f}".format(values3[i]), ha='center', va='bottom',fontsize=12, fontweight='semibold')



plt.tight_layout()

# 显示图形
plt.show()

