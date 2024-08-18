import torch
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import argparse

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
    embedding = torch.load('Washed_Embed/Result_Embed/{}/{}/{}.pt'.format(dataset,name_without_epoch, name)).to(device)
    category = pd.read_csv('Washed/{}/category.csv'.format(poi_model_name), usecols=['geo_id', 'category'])

    inputs=category.geo_id.to_numpy()
    labels=category.category.to_numpy()
    num_class = labels.max()+1

    node_embedding=embedding[torch.tensor(inputs)].cpu()

    print(f'Start Kmeans, data.shape = {node_embedding.shape}, kinds = {num_class}')
    k_means = KMeans(n_clusters=num_class, random_state=42)
    k_means.fit(node_embedding)
    y_predict = k_means.predict(node_embedding)
    y_predict_useful = y_predict
    nmi = metrics.normalized_mutual_info_score(labels, y_predict_useful)

    result = pd.DataFrame({
        'name': args.NAME,
        'nmi': nmi,
    }, index=[1])



    import os
    save_path = './Washed_Result_Metric/' + args.dataset + '/' + name +'/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    if args.save_path != 'train':
        result.to_csv(save_path + name + '.cluster', index=False)


    

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(node_embedding)
    data_pca = pca.transform(node_embedding)
    data_pca = pd.DataFrame(data_pca)
    data_pca.insert(data_pca.shape[1], 'labels', labs)

    cluster_list = []
    for n in range(num_class):
        cluster = node_embedding[k_means.lables_ == n]
        cluster.append(cluster)

    predicted_labels = y_predict

    import matplotlib.pyplot as plt
    plt.scatter(data_pca.shape[:, 0], data_pca.shape[:, 1], c=predicted_labels, s=50, cmap='viridis')

