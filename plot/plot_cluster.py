import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import cm

# 准备数据
font_w = 24
model = ['Skip-Gram', 'POI2Vec', 'Geo-Teaser', 'TALE', 'Hier','CTLE']
model_enhanced = ['Skip-Gram+', 'POI2Vec+', 'Geo-Teaser+', 'TALE+', 'Hier+', 'CTLE+']
values_ny = [0.232148, 0.220677, 0.174622, 0.049854, 0.20929,0.220738]
values_ny_enhance = [0.649379398673667, 0.703512104145506, 0.64653, 0.655357, 0.710924610165569, 0.648776]
values_sg = [0.205126, 0.175793, 0.181098, 0.046599, 0.160105,0.155127]
values_sg_enhance = [0.664172207664908, 0.659840775276634, 0.663819740369317, 0.656248167747075, 0.645804162103704, 0.653283424681911]
values_tky = [0.22618, 0.220407, 0.205277, 0.066118, 0.203471,0.217009]
values_tky_enhance = [0.413264500300939,0.659841,0.518985104501953,0.409118652243554,0.415813105694656,0.575112034544004]
index = range(len(model))
cmap_name = 'hsv'

cmap = cm.get_cmap(cmap_name)

colors = ['#76E0D6','#26CDD5', '#318EDE', '#2D59C6','#1F4E79','#002060']
barwidth=0.7
fig, axs = plt.subplots(1, 3, figsize=(20, 10))

axs[0].bar(model, values_ny, color=colors, label=model,width=barwidth)
axs[0].bar(model_enhanced, values_ny_enhance, color=colors, label=model_enhanced,width=barwidth)
axs[0].set_title('New York',fontsize=font_w,fontweight='bold')
axs[0].set_xticklabels(model+model_enhanced,rotation=90,fontweight='semibold',fontsize=font_w)
axs[0].tick_params(axis='y', labelsize=font_w)
axs[0].set_ylabel('NMI', fontsize=font_w-4, fontweight='semibold')



axs[1].bar(model, values_sg, color=colors, label=model,width=barwidth)
axs[1].bar(model_enhanced, values_sg_enhance, color=colors, label=model_enhanced,width=barwidth)
axs[1].set_title('Singapore',fontsize=font_w,fontweight='bold')
axs[1].set_xticklabels(model+model_enhanced,rotation=90,fontweight='semibold',fontsize=font_w)
axs[1].set_yticklabels([])






axs[2].bar(model, values_tky, color=colors, label=model,width=barwidth)
axs[2].bar(model_enhanced, values_tky_enhance, color=colors, label=model_enhanced,width=barwidth)
axs[2].set_title('Tokyo',fontsize=font_w,fontweight='bold')
axs[2].set_xticklabels(model+model_enhanced,rotation=90,fontweight='semibold',fontsize=font_w)
axs[2].set_yticklabels([])






plt.tight_layout()

# 显示图形
plt.show()

