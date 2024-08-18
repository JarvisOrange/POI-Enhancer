from torch import nn
from torch.nn import init
import numpy as np
import pandas as pd
from itertools import zip_longest
import torch
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader	
import os


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
        
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.Embedding):
        embed_size = m.weight.size(-1)
        if embed_size > 0:
            init_range = 0.5/m.weight.size(-1)
            init.uniform_(m.weight.data, -init_range, init_range)



class PoiDataset(data.Dataset):
    def __init__(self,  path, device):
        df = pd.read_csv(path,sep=',', header=0, usecols=['geo_id_washed','type'])
        df = df[df['type']=='Point']
        df = df.drop(['type'], axis=1)
        first = df.iloc[0,0]
        df['geo_id_washed'] = df['geo_id_washed'].apply(lambda x: x - first)

        self.device = device
        self.data = df
        self.data = self.data.values

        
    def __getitem__(self, index):
        data = self.data[index]
        data = torch.IntTensor(data).to(self.device)
        return data
    
    def __len__(self):
        return len(self.data)
    
class ContrastDataset(data.Dataset):
    def __init__(self,  path, device, simple=False,ablation=0):
        df = pd.read_csv(path,sep=',', header=0, dtype={'anchor':int,'positive':int, 'negative':str})
        if ablation == 3:
            df= df.sample(frac=0.2)
            
        if simple == 'True':
                df= df.sample(frac=0.0001)
        
        df['negative'] = df['negative'].apply(lambda x : eval(x))
        self.device = device
        self.data = df
        self.data = self.data.values

        
    def __getitem__(self, index):
        anchor, pos, negative = self.data[index]
        data = [anchor, pos] + negative
        data = torch.IntTensor(data).to(self.device)
        
        return data
    
    def __len__(self):
        return len(self.data)
    




def simloss(embed_new, st_embed):
    x1 = F.cosine_similarity(embed_new.unsqueeze(1), embed_new.unsqueeze(0), dim=2)
    x2 = F.cosine_similarity(st_embed.unsqueeze(1), st_embed.unsqueeze(0), dim=2)
    loss = F.mse_loss(x1, x2)
    
    return loss   

    

def save_embed(Model, dataset, LLM, dim, poi_model, epoch, device, align_layer_num=3, cross_layer_num=3, ablation=0):
    if ablation != 0:
        name_embed = dataset + '_' + LLM + '_' + poi_model + '_'+ str(dim) + '_Epoch_' + str(epoch) +'_ablation' + str(ablation)+'.pt'

        name_statedict = dataset + '_' + LLM + '_' + poi_model + '_'+ str(dim) +  '_Epoch_' +str(epoch) +'_ablation'+ str(ablation)+'.pt'
    else:
        if align_layer_num != 3:
            name_embed = dataset + '_' + LLM + '_' + poi_model + '_'+ str(dim) + '_Epoch_' + str(epoch) +'_align_' + str(align_layer_num) +'.pt'

        elif cross_layer_num != 3:
            name_embed = dataset + '_' + LLM + '_' + poi_model + '_'+ str(dim) + '_Epoch_' + str(epoch) +'_cross_' + str(cross_layer_num) +'.pt'

        else: 
            name_embed = dataset + '_' + LLM + '_' + poi_model + '_'+ str(dim) + '_Epoch_' + str(epoch) +'.pt'
            
        
    
    if ablation != 0:
        embed_path = './Washed_Embed/Ablation_Embed/'+ dataset +'/' + dataset + '_' + LLM + '_' + poi_model + '_'+ str(dim) + '/'
    elif align_layer_num != 3 or cross_layer_num != 3:
        embed_path = './Washed_Embed/Para_Embed/'+ dataset +'/' + dataset + '_' + LLM + '_' + poi_model + '_'+ str(dim) + '/'
    else:
        embed_path = './Washed_Embed/Result_Embed/'+ dataset +'/' + dataset + '_' + LLM + '_' + poi_model + '_'+ str(dim) + '/'

    # model_path =  "./Washed_Model_state_dict_cache/" + dataset  +'/'+ dataset + '_' + LLM + '_' + poi_model + '_'+ str(dim) + '/'

    if not os.path.exists(embed_path):
        os.makedirs(embed_path)


    batch_size = 128


    # torch.save({'model': Model.state_dict()}, model_path + name_statedict)


    path = './Dataset/' + dataset +'/'+ dataset.lower() + '_geo.csv'
    poi_dataset = PoiDataset(path, device)
    poi_dataloader = DataLoader(poi_dataset, batch_size = batch_size, shuffle = False)

    

    result_embed = torch.empty((len(poi_dataset), dim)).cpu()

    # torch.cuda.empty_cache()
    index = 0
    with torch.no_grad():
        for step, batch in enumerate(poi_dataloader):
            try:
                out = Model(batch)
            except:
                print(batch)
            out = out[0].squeeze(1)


            if out.shape[0] !=  batch_size:
                result_embed[step * batch_size :step * batch_size + out.shape[0],:] = out.cpu()
                index +=  out.shape[0]
            else:
                result_embed[step * batch_size : (step +1)* batch_size,:] = out.cpu()
                index +=  batch_size

    torch.save(result_embed, embed_path + name_embed)