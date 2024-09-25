import random
import pandas as pd
import torch
from torch_geometric.utils import structured_negative_sampling

# Load user nodes
def load_node_csv(path, index_col, min_interactions=1):
    df = pd.read_csv(path, index_col=index_col)
    user_interactions = df.groupby(index_col).size()
    valid_users = user_interactions[user_interactions >= min_interactions].index
    df = df[df.index.isin(valid_users)]
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    return mapping, df

# Load item nodes
def load_item_csv(path, index_col, rating_df, min_interactions=1):
    df = pd.read_csv(path, index_col=index_col)
    item_interactions = rating_df.groupby('itemId').size()
    valid_items = item_interactions[item_interactions >= min_interactions].index
    df = df[df.index.isin(valid_items)]
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    return mapping, df

# Merge filtered users from two files
def load_user_csv_total(rating_path1, rating_path2, index_col, min_interactions=1):
    users1 = load_filtered_users(rating_path1, index_col, min_interactions)
    users2 = load_filtered_users(rating_path2, index_col, min_interactions)
    all_users = list(set(users1).union(set(users2)))
    total_mapping = {user: i for i, user in enumerate(all_users)}
    return total_mapping

# Merge filtered items from two files
def load_item_csv_total(item_path1, item_path2, rating_path1, rating_path2, index_col, min_interactions=1):
    items1 = load_filtered_items(item_path1, rating_path1, index_col, min_interactions)
    items2 = load_filtered_items(item_path2, rating_path2, index_col, min_interactions)
    all_items = list(set(items1).union(set(items2)))
    total_mapping = {item: i for i, item in enumerate(all_items)}
    return total_mapping

def load_filtered_items(item_path, rating_path, index_col, min_interactions=1):
    df_items = pd.read_csv(item_path, index_col=index_col)
    df_ratings = pd.read_csv(rating_path)
    item_interactions = df_ratings.groupby(index_col).size()
    valid_items = item_interactions[item_interactions >= min_interactions].index
    df_items_filtered = df_items[df_items.index.isin(valid_items)]
    return df_items_filtered.index.unique()

def load_filtered_users(rating_path, index_col, min_interactions=1):
    df_ratings = pd.read_csv(rating_path)
    user_interactions = df_ratings.groupby(index_col).size()
    valid_users = user_interactions[user_interactions >= min_interactions].index
    return valid_users

# Load edges (user-item interactions)
def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping, link_index_col, rating_threshold=4):
    df = pd.read_csv(path)
    df = df[df[src_index_col].isin(src_mapping.keys()) & df[dst_index_col].isin(dst_mapping.keys())]
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_attr = torch.from_numpy(df[link_index_col].values).view(-1, 1).to(torch.long) >= rating_threshold

    edge_index = [[], []]
    for i in range(edge_attr.shape[0]):
        if edge_attr[i]:
            edge_index[0].append(src[i])
            edge_index[1].append(dst[i])

    return torch.tensor(edge_index)

def sample_mini_batch(batch_size, edge_index, num_items):
    user_indices = torch.empty(0, dtype=torch.long)
    pos_item_indices = torch.empty(0, dtype=torch.long)
    neg_item_indices = torch.empty(0, dtype=torch.long)
    while user_indices.size(0) < batch_size:
        edges = structured_negative_sampling(edge_index)
        edges = torch.stack(edges, dim=0)
        indices = random.choices([i for i in range(edges[0].shape[0])], k=batch_size)
        batch = edges[:, indices]

        user_indices_batch, pos_item_indices_batch, neg_item_indices_batch = batch[0], batch[1], batch[2]
        valid_neg_indices = neg_item_indices_batch < num_items

        user_indices_batch = user_indices_batch[valid_neg_indices]
        pos_item_indices_batch = pos_item_indices_batch[valid_neg_indices]
        neg_item_indices_batch = neg_item_indices_batch[valid_neg_indices]

        user_indices = torch.cat([user_indices, user_indices_batch], dim=0)
        pos_item_indices = torch.cat([pos_item_indices, pos_item_indices_batch], dim=0)
        neg_item_indices = torch.cat([neg_item_indices, neg_item_indices_batch], dim=0)

    user_indices = user_indices[:batch_size]
    pos_item_indices = pos_item_indices[:batch_size]
    neg_item_indices = neg_item_indices[:batch_size]

    return user_indices, pos_item_indices, neg_item_indices