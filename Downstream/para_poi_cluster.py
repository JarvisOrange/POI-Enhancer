import torch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
import argparse
from sklearn.preprocessing import StandardScaler

device='cuda:1'
def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1, help="gpu")

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
        "--save_path",
        type=str,
        default=None,
        help="Result save path",
    )

    parser.add_argument(
        "--origin",
        type=str,
        default=None,
        
    )


    args = parser.parse_args()

    return args

if __name__ == '__main__':    
    args = create_args()
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    name = args.NAME
    dataset = args.dataset
    poi_model_name = args.POI_MODEL_NAME

    temp = name.split('_')
    name_without_epoch = '_'.join(temp[:4])
    

    
    
    if args.origin == '1':
        embedding = torch.load('Washed/skipgram_256_ny/poi_repr.pth').to(device)
    else:
        embedding = torch.load('Washed_Embed/Result_Embed/{}/{}/{}.pt'.format(dataset,name_without_epoch, name)).to(device)
    category = pd.read_csv('Washed/{}/category.csv'.format(poi_model_name), usecols=['geo_id', 'category'])

    inputs=category.geo_id.to_numpy()
    labels=category.category.to_numpy()

    

    from collections import Counter

    result = Counter(list(labels))
   
    d = sorted(result.items(), key=lambda x: x[1], reverse=True)

    top_ten_values = [x[0] for x in d[0:5]]


    num_class = labels.max()+1

    node_embedding=embedding[torch.tensor(inputs)].cpu()

    node_embedding = StandardScaler().fit_transform(node_embedding)

    print(f'Start Kmeans, data.shape = {node_embedding.shape}, kinds = {num_class}')
    k_means = KMeans(n_clusters=num_class, random_state=42)
    k_means.fit(node_embedding)
    y_predict = k_means.predict(node_embedding)
    y_predict_useful = y_predict
    
    

    
    pca = PCA(n_components=2)
    pca.fit(node_embedding)
    data_pca = pca.transform(node_embedding)
    data_pca = pd.DataFrame(data_pca)


    data_pca.insert(data_pca.shape[1], 'labels', labels)
    
    cluster_list = data_pca[data_pca['labels'].isin(top_ten_values)]


    predicted_labels = y_predict

    import matplotlib.pyplot as plt

    print(len(cluster_list))

    data_pca = pd.DataFrame(cluster_list)



    plt.scatter(data_pca.iloc[:, 0], data_pca.iloc[:, 1], c=data_pca.iloc[:, 2], s=20, cmap='viridis')

    
plt.savefig(args.origin+".jpg")

