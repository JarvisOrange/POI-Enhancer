import torch
from torch import nn
from libcity_utils import weight_init, next_batch
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import pandas as pd

import argparse
import os

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="gpu")

    parser.add_argument(
        "--NAME",
        type=str
    )

    parser.add_argument(
        "--POI_MODEL_NAME",
        type=str
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="NY",
        choices=["NY","SG","TKY"],
        help="which dataset",
    )

    parser.add_argument(
        "--embed_size",
        type=int,
        default=256,
        choices=[128, 256],
        help="The embedding size",
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help='The ratio of test set in the whole dataset'
    )

    parser.add_argument(
        "--flow_len",
        type=int,
        default=6,
        help='The length of flow data in the dataset'
    )

    parser.add_argument(
        "--prompt",
        type=int,
        default=0,
        help='0 means address, 1 means address+visit, 2 means address+surrounding, 3means all sum'
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Result save path",
    )

    parser.add_argument(
        "--ablation",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--para",
        type=str,
        default=None,
    )

    

    args = parser.parse_args()

    return args

# Training code adapted frim TALE
class Seq2seqFlowPredictor(nn.Module):
    def __init__(self, loc_embed_layer, loc_embed_size, fc_size, hidden_size, num_layers):
        super().__init__()

        self.loc_embed_layer = loc_embed_layer

        _rnn_input_size = fc_size * 2 + 16
        self.recent_encoder = nn.GRU(input_size=_rnn_input_size, hidden_size=hidden_size, num_layers=num_layers,
                                     batch_first=True, dropout=0.1)
        self.encoder_merge_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.decoder = nn.GRU(input_size=_rnn_input_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, dropout=0.1)
        self.time_embed = nn.Embedding(24, 16)

        self.embed_linear = nn.Linear(loc_embed_size, fc_size)
        self.flow_linear = nn.Linear(1, fc_size)
        self.out_linear = nn.Linear(hidden_size, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

        self.apply(weight_init)

    def forward(self, recent_history, loc_index, recent_hour, **kwargs):
        """
        :param recent_history: historical flow sequences, shape (batch_size, his_len)
        :param remote_history: remote historical flow sequences from previews day, shape (batch_size, his_len)
        :param target_seq: target flow sequences, shape (batch_size, pre_len)
        :param loc_index: location indices corresponding to flow sequences, shape (batch_size)
        :param recent_hour: hour indices of all sequences, shape (batch_size, his_len+pre_len)
        :return: prediction of future flows, shape (batch_size, pre_len)
        """

        
        loc_embed = self.loc_embed_layer[loc_index]
        loc_h = self.embed_linear(loc_embed)  # (batch_size, latent_size)
        recent_flow_h = self.flow_linear(recent_history.unsqueeze(-1))  # (batch_size, his_len+pre_len, latent_size)
        recent_hour_embed = self.time_embed(recent_hour)  # (batch_size, his_len+pre_len, time_embed_size)
        recent_cat_h = torch.cat([recent_flow_h, loc_h.unsqueeze(1).repeat(1, recent_flow_h.size(1), 1), recent_hour_embed], -1)  # (batch_size, his_len+pre_len, rnn_input_size)
        recent_cat_h = self.dropout(self.tanh(recent_cat_h))

        recent_encoder_out, recent_hc = self.recent_encoder(recent_cat_h)
        decoder_out, hc = self.decoder(recent_cat_h[:, -1:], recent_hc)  # (batch_size, pre_len, hidden_size)
        out = self.out_linear(decoder_out).squeeze(-1)
        return out

def train_flow_predictor(pre_model, train_set, test_set, batch_size, num_epoch,
                         lr, device, flow_len, **kwargs):
    pre_model = pre_model.to(device)
    optimizer = torch.optim.Adam(pre_model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    scaler = StandardScaler()

    def _pre_batch(input_model, batch, scaler):
        poi_index, recent_hour, recent_seq = zip(*batch)
        poi_index = torch.tensor(poi_index).long().to(device)

        with torch.no_grad():
            recent_hour = torch.stack([torch.arange(hour, hour + flow_len - 1) for hour in recent_hour]).long().to(device) % 24
        with torch.no_grad():
            recent_hour = recent_hour % 24
        recent_seq = scaler.transform(recent_seq)
        recent_seq = torch.tensor(recent_seq).float().to(device)

        recent_history, label = recent_seq[:, :-1], recent_seq[:, -1]
        pre = input_model(recent_history=recent_history, loc_index=poi_index, recent_hour=recent_hour)
        return pre.squeeze(), label

    def _test_epoch(input_model, input_set, scaler):
        input_model.eval()
        pres, labels = [], []
        for batch in next_batch(input_set, batch_size=256):
            pre, label = _pre_batch(input_model, batch, scaler)
            pres.append(pre.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
        pres, labels = (np.concatenate(item) for item in (pres, labels))
        return pres, labels

    scaler.fit(np.stack([data[2] for data in train_set]))
    min_mae = 1e6
    best_tuple = None

    for epoch in range(num_epoch):
        for batch in next_batch(shuffle(train_set), batch_size):
            pre_model.train()
            pre, label = _pre_batch(pre_model, batch, scaler)
            optimizer.zero_grad()
            loss = loss_func(pre, label)
            loss.backward()
            optimizer.step()

        pres, labels = _test_epoch(pre_model, test_set, scaler)
        mae, mse, mape = mean_absolute_error(labels, pres), mean_squared_error(labels, pres), \
                            mean_absolute_percentage_error(labels, pres)
        rmse = np.sqrt(mse)
        print(f'Epoch {epoch}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE:{mape:.4f}.')
        if mae < min_mae:
            min_mae = mae
            best_tuple = (mae, rmse, mape)

    return best_tuple

if __name__ == '__main__':
    args = create_args()
    dataset = args.dataset
    name = args.NAME
    poi_model_name = args.POI_MODEL_NAME
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    print(device)

    if args.prompt == 0:
        name = str(dataset)+'_' + args.NAME +'_address_LAST'
        if args.NAME != 'Llama2':
            poi_embedding = torch.load('Washed_Embed/LLM_Embed/{}/{}/{}.pt'.format(args.NAME, dataset, name)).to(device)
        else:
            poi_embedding = torch.load('Washed_Embed/LLM_Embed/{}/{}.pt'.format(dataset, name)).to(device)
    elif args.prompt == 1:
        name  = str(dataset)+'_llama2_address_time_LAST'
        name1 = str(dataset)+'_llama2_address_LAST'
        poi_embedding1 = torch.load('Washed_Embed/LLM_Embed/{}/{}.pt'.format(dataset, name1)).to(device)
        name2 = str(dataset)+'_llama2_time_LAST'
        poi_embedding2 = torch.load('Washed_Embed/LLM_Embed/{}/{}.pt'.format(dataset, name2)).to(device)

        poi_embedding1.requires_grad=False
        poi_embedding2.requires_grad=False

        poi_embedding = poi_embedding1 + poi_embedding2
    elif args.prompt == 2:
        name  = str(dataset)+'_llama2_address_cat_LAST'
        name1 = str(dataset)+'_llama2_address_LAST'
        poi_embedding1 = torch.load('Washed_Embed/LLM_Embed/{}/{}.pt'.format(dataset, name1)).to(device)
        name2 = str(dataset)+'_llama2_cat_nearby_LAST'
        poi_embedding2 = torch.load('Washed_Embed/LLM_Embed/{}/{}.pt'.format(dataset, name2)).to(device)

        poi_embedding1.requires_grad=False
        poi_embedding2.requires_grad=False

        poi_embedding = poi_embedding1 + poi_embedding2
    elif args.prompt == 3:
        name  = str(dataset)+'_llama2_all_LAST'
        name1 = str(dataset)+'_llama2_address_LAST'
        poi_embedding1 = torch.load('Washed_Embed/LLM_Embed/{}/{}.pt'.format(dataset, name1)).to(device)
        name2 = str(dataset)+'_llama2_time_LAST'
        poi_embedding2 = torch.load('Washed_Embed/LLM_Embed/{}/{}.pt'.format(dataset, name2)).to(device)
        name3 = str(dataset)+'_llama2_cat_nearby_LAST'
        poi_embedding3 = torch.load('Washed_Embed/LLM_Embed/{}/{}.pt'.format(dataset, name3)).to(device)

        poi_embedding1.requires_grad=False
        poi_embedding2.requires_grad=False
        poi_embedding3.requires_grad=False

        poi_embedding = poi_embedding1 + poi_embedding2 + poi_embedding3


    poi_embedding.requires_grad = False
    print(poi_embedding.shape)
    with torch.no_grad():
        if args.NAME == 'gpt2':
            avg_pool = nn.AvgPool1d(kernel_size=3, stride=3)
            poi_embedding = poi_embedding.unsqueeze(1)
            poi_embedding = avg_pool(poi_embedding)
            poi_embedding = poi_embedding.squeeze(1)
        else:
            avg_pool = nn.AvgPool1d(kernel_size=16, stride=16)
            poi_embedding = poi_embedding.unsqueeze(1)
            poi_embedding = avg_pool(poi_embedding)
            poi_embedding = poi_embedding.squeeze(1)

    poi_embedding.requires_grad = True
    

    poi_embedding = poi_embedding.to(torch.float)
    


    # embedding = torch.load('Washed/poi2vec_256_sg/poi_repr.pth').to(device)
    dataset = torch.load('Washed/common/{}_flow.pth'.format(dataset.lower()))
    batch_size = 128

    

    np.random.shuffle(dataset)
    train_set = dataset[int(args.test_ratio * len(dataset)):]
    test_set = dataset[:int(args.test_ratio * len(dataset))]

    model = Seq2seqFlowPredictor(loc_embed_layer=poi_embedding, loc_embed_size=256, fc_size=256,
                                  hidden_size=512, num_layers=2)

    best_mae, best_rmse, best_mape = train_flow_predictor(model, train_set, test_set, batch_size, num_epoch=100,
                               lr=1e-4, early_stopping_round=10, device=device, flow_len=args.flow_len)
    print(f'Best epoch: MAE: {best_mae:.6f}, RMSE: {best_rmse:.6f}, MAPE: {best_mape:.6f}')

    import os
    save_path = './Washed_Result_Metric/' + args.dataset + '/' + name +'/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    if args.save_path != 'train':
        pd.DataFrame({
            'mae': best_mae,
            'mape': best_mape,
            'rmse': best_rmse
        }, index=[1]).to_csv(save_path + name + '.flow')