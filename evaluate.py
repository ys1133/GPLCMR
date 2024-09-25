import torch
import numpy as np
import torch.nn.functional as F

def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final, neg_items_emb_0,
             lambda_val):
    reg_loss = lambda_val * (users_emb_0.norm(2).pow(2) +
                             pos_items_emb_0.norm(2).pow(2) +
                             neg_items_emb_0.norm(2).pow(2))

    pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
    pos_scores = torch.sum(pos_scores, dim=-1)

    neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
    neg_scores = torch.sum(neg_scores, dim=-1)

    loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss
    return loss

def cross_entropy_loss(users_emb_final, pos_items_emb_final, neg_items_emb_final):
    pos_scores = torch.mul(users_emb_final, pos_items_emb_final).sum(dim=-1)
    neg_scores = torch.mul(users_emb_final, neg_items_emb_final).sum(dim=-1)
    targets = torch.ones_like(pos_scores)
    logits = torch.cat([pos_scores.unsqueeze(-1), neg_scores.unsqueeze(-1)], dim=-1)
    loss = torch.nn.functional.cross_entropy(logits, targets.long())
    return loss

# Knowledge distillation loss using KL divergence
def distillation_loss(student_output, teacher_output, temperature=3.0):
    student_output = F.log_softmax(student_output / temperature, dim=1)
    teacher_output = F.softmax(teacher_output / temperature, dim=1)
    return F.kl_div(student_output, teacher_output, reduction='batchmean') * (temperature ** 2)

# Get positive items for each user
def get_user_positive_items(edge_index):
    user_pos_items = {}
    for i in range(edge_index.shape[1]):
        user = edge_index[0][i].item()
        item = edge_index[1][i].item()
        if user not in user_pos_items:
            user_pos_items[user] = []
        user_pos_items[user].append(item)
    return user_pos_items

# Calculate recall, precision, and hit rate at K
def RecallPrecision_ATk(groundTruth, r, k):
    num_correct_pred = torch.sum(r, dim=-1)
    user_num_liked = torch.Tensor([len(groundTruth[i]) for i in range(len(groundTruth))])

    recall = torch.mean(num_correct_pred / user_num_liked)
    precision = torch.mean(num_correct_pred) / k
    hit_rate = torch.mean((num_correct_pred > 0).float())
    return recall.item(), precision.item(), hit_rate.item()

# Calculate Normalized Discounted Cumulative Gain (NDCG) at K
def NDCGatK_r(groundTruth, r, k):
    assert len(r) == len(groundTruth)
    test_matrix = torch.zeros((len(r), k), dtype=torch.float)

    for i, items in enumerate(groundTruth):
        length = min(len(items), k)
        test_matrix[i, :length] = 1

    max_r = test_matrix
    idcg = torch.sum(max_r * 1. / torch.log2(torch.arange(2, k + 2, dtype=torch.float)), axis=1)

    dcg = r * (1. / torch.log2(torch.arange(2, k + 2, dtype=torch.float)))
    dcg = torch.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.

    ndcg = dcg / idcg
    ndcg[torch.isnan(ndcg)] = 0.
    return torch.mean(ndcg).item()

# Get metrics like recall, precision, hit rate, and NDCG
def get_metrics(model, edge_index, exclude_edge_indices, k):
    user_embedding = model.users_emb.weight
    item_embedding = model.items_emb.weight

    rating = torch.matmul(user_embedding, item_embedding.T)

    for exclude_edge_index in exclude_edge_indices:
        user_pos_items = get_user_positive_items(exclude_edge_index)

        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)

        rating[exclude_users, exclude_items] = -(1 << 10)

    _, top_K_items = torch.topk(rating, k=k)

    users = edge_index[0].unique()
    test_user_pos_items = get_user_positive_items(edge_index)

    test_user_pos_items_list = [test_user_pos_items[user.item()] for user in users]

    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    recall, precision, hit_rate = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)

    return recall, precision, hit_rate, ndcg

def get_metrics2(model, edge_index, exclude_edge_indices, k,users_emb_final2_eval,items_emb_final2_eval):
    user_embedding = users_emb_final2_eval
    item_embedding = items_emb_final2_eval
    rating = torch.matmul(user_embedding, item_embedding.T)

    for exclude_edge_index in exclude_edge_indices:
        user_pos_items = get_user_positive_items(exclude_edge_index)

        exclude_users = []
        exclude_items = []
        for user, items in user_pos_items.items():
            exclude_users.extend([user] * len(items))
            exclude_items.extend(items)
        rating[exclude_users, exclude_items] = -(1 << 10)


    _, top_K_items = torch.topk(rating, k=k)
    users = edge_index[0].unique()

    test_user_pos_items = get_user_positive_items(edge_index)
    test_user_pos_items_list = [
        test_user_pos_items[user.item()] for user in users]

    r = []
    for user in users:
        ground_truth_items = test_user_pos_items[user.item()]
        label = list(map(lambda x: x in ground_truth_items, top_K_items[user]))
        r.append(label)
    r = torch.Tensor(np.array(r).astype('float'))

    recall, precision,hit_rate = RecallPrecision_ATk(test_user_pos_items_list, r, k)
    ndcg = NDCGatK_r(test_user_pos_items_list, r, k)

    return recall, precision,hit_rate, ndcg

# Evaluation with BPR loss and metric calculation
def evaluation(model, edge_index, sparse_edge_index, exclude_edge_indices, k, lambda_val, num_users, num_items,
               prompt=None):
    users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(sparse_edge_index, num_users, num_items,prompt)
    edges = custom_negative_sampling(edge_index, num_items)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]

    users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
    pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
    neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]

    loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final, pos_items_emb_0, neg_items_emb_final,
                    neg_items_emb_0, lambda_val)

    recall, precision, hit_rate, ndcg = get_metrics(model, edge_index, exclude_edge_indices, k)
    return loss, recall, precision, hit_rate, ndcg

def evaluation2(model, edge_index, exclude_edge_indices, k, num_items, users_emb_final2_eval,items_emb_final2_eval):
    edges = custom_negative_sampling(edge_index,num_items)
    user_indices, pos_item_indices, neg_item_indices = edges[0], edges[1], edges[2]
    neg_item_indices = neg_item_indices[neg_item_indices < num_items]

    users_emb_final = users_emb_final2_eval[user_indices]
    pos_items_emb_final = items_emb_final2_eval[pos_item_indices]
    neg_items_emb_final= items_emb_final2_eval[neg_item_indices]

    loss = cross_entropy_loss(users_emb_final, pos_items_emb_final, neg_items_emb_final)

    recall, precision, hit_rate, ndcg = get_metrics2(
        model, edge_index, exclude_edge_indices, k, users_emb_final2_eval, items_emb_final2_eval)

    return loss, recall, precision,hit_rate, ndcg

# Custom negative sampling to generate negative item examples
def custom_negative_sampling(edge_index, num_items):
    num_edges = edge_index.size(1)
    user_indices = edge_index[0]
    pos_item_indices = edge_index[1]
    user_pos_items = get_user_positive_items(edge_index)
    neg_item_indices = []

    for user in user_indices.tolist():
        neg_item = torch.randint(0, num_items, (1,)).item()
        while neg_item in user_pos_items.get(user, []):
            neg_item = torch.randint(0, num_items, (1,)).item()
        neg_item_indices.append(neg_item)

    neg_item_indices = torch.tensor(neg_item_indices)
    return user_indices, pos_item_indices, neg_item_indices


