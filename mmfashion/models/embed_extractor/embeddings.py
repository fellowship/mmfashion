import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmfashion.models import build_fashion_recommender
from mmfashion.utils import get_img_tensor
from sklearn.decomposition import PCA

def get_images(use_cuda):
    import os

    img_tensors = []
    item_ids = []
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'dockers_pics')
    files = [name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))]
    for f_name in files:
        if f_name == "wget_links.sh":
            continue
        path = os.path.join(data_dir, f_name)
        try:
            tensor = get_img_tensor(path, use_cuda)
            item_ids.append(f_name.split('%')[0])  # ignore the %0D suffix that gets attached with wget for each image
            img_tensors.append(tensor)
        except:
            print("Falied to get: ", path)
    img_tensors = torch.cat(img_tensors)
    return img_tensors, item_ids


def get_mmfashion_embeddings(model, img_tensors, image_ids):
    """
    Use pretrained mmfashion model to get embeddings for each image tensor
    :param model: mmfashion model with ResNet18 base
    :param img_tensors: list of pytorch image tensors
    :param image_ids:  list of image urls for each tensor
    :return:
    """
    pca = PCA(n_components=10)
    embeds = []
    embedding_dict = {}
    with torch.no_grad():
        batches = torch.split(img_tensors, 10)
        for batch in batches:
            embed = model(batch, return_loss=False)
            embeds.append(embed.data.cpu())
    embeds = torch.cat(embeds)
    for idx, embed in enumerate(embeds):
        pca.fit(embed)
        embedding_dict[image_ids[idx]] = pca.singular_values_
        print(pca.singular_values_)
    return embedding_dict


def get_embeddings_dict(products_df):
    """
    Use pretrained type aware recommendations model(by mmfashion) to generate embeddings for each product in products_df


    :param products_df: pd.DataFrame containing product details
    :return: embeddings dict that maps product image url to mmfashion embedding
    """

    cfg = Config.fromfile("configs/fashion_recommendation/type_aware_recommendation_polyvore_disjoint_l2_embed.py")
    cfg.load_from = "checkpoint/FashionRecommend/TypeAware/disjoint/l2_embed/epoch_16.pth"

    # create model
    model = build_fashion_recommender(cfg.model)
    load_checkpoint(model, cfg.load_from, map_location='cpu')
    print('load checkpoint from: {}'.format(cfg.load_from))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    model.eval()

    img_tensors, image_ids = get_images(use_cuda)
    embeddings_dict = get_mmfashion_embeddings(model, img_tensors, image_ids)

    return embeddings_dict


def main():
    import pandas as pd
    import os
    import pickle
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data')
    input_path = os.path.join(data_dir, 'sample_data.csv')
    output_path = os.path.join(data_dir, 'image_embeddings2.txt')
    products_df = pd.read_csv(input_path)

    # download_images(products_df)

    with open(output_path, 'wb') as f:
        pickle.dump(get_embeddings_dict(products_df), f)


if __name__ == '__main__':
    main()
