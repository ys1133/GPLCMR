import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch_sparse import SparseTensor

from model import LightGCN
from data import load_node_csv, load_item_csv, load_edge_csv,load_item_csv_total,load_user_csv_total,sample_mini_batch
from evaluate import bpr_loss, cross_entropy_loss,evaluation,evaluation2,distillation_loss
from prompt import GPFplusAtt
from config import config


def run(config):
    # Load user and item data from two markets
    rating_path=config['rating_path']
    item_path= config['item_path']
    rating_path2=config['rating_path2']
    item_path2=config['item_path2']

    # Combine user and item mappings from two markets
    user_mapping = load_user_csv_total(rating_path,rating_path2, index_col='userId')
    item_mapping = load_item_csv_total(item_path,item_path2,rating_path,rating_path2, index_col='itemId')

    # Load user and item data for each market separately
    user_mapping1, filtered_user_df1  = load_node_csv(rating_path, index_col='userId')
    item_mapping1, filtered_item_df1 = load_item_csv(item_path, index_col='itemId', rating_df=filtered_user_df1)
    user_mapping2 ,filtered_user_df2 = load_node_csv(rating_path2, index_col='userId')
    item_mapping2, filtered_item_df2 = load_item_csv(item_path2,index_col='itemId', rating_df=filtered_user_df2)

    # Load interaction data between users and items
    edge_index1 = load_edge_csv(
        rating_path,
        src_index_col='userId',
        src_mapping=user_mapping1,
        dst_index_col='itemId',
        dst_mapping=item_mapping1,
        link_index_col='rate',
        rating_threshold=4,
    )
    edge_index2 = load_edge_csv(
        rating_path2,
        src_index_col='userId',
        src_mapping=user_mapping2,
        dst_index_col='itemId',
        dst_mapping=item_mapping2,
        link_index_col='rate',
        rating_threshold=4,
    )
    edge_index = torch.cat([edge_index1, edge_index2], dim=1)

    # Split the interaction data into training, validation, and test sets
    num_users, num_items = len(user_mapping), len(item_mapping)
    num_interactions = edge_index.shape[1]
    all_indices = [i for i in range(num_interactions)]

    train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=1)
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=1)

    train_edge_index = edge_index[:, train_indices]
    val_edge_index = edge_index[:, val_indices]
    test_edge_index = edge_index[:, test_indices]

    # Initialize device and sparse tensors for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Pretraining - Using device {device}.")

    train_sparse_edge_index = SparseTensor(
        row=train_edge_index[0].to(device),
        col=train_edge_index[1].to(device),
        sparse_sizes=(num_users + num_items, num_users + num_items)
    )
    val_sparse_edge_index = SparseTensor(
        row=val_edge_index[0].to(device),
        col=val_edge_index[1].to(device),
        sparse_sizes=(num_users + num_items, num_users + num_items)
    )
    # test_sparse_edge_index = SparseTensor(
    #     row=test_edge_index[0].to(device),
    #     col=test_edge_index[1].to(device),
    #     sparse_sizes=(num_users + num_items, num_users + num_items)
    # )

    # Initialize model
    model = LightGCN(num_users, num_items)
    # define contants
    ITERATIONS = config['iterations']
    BATCH_SIZE = config['batch_size']
    LR = config['lr']
    ITERS_PER_EVAL = config['iters_per_eval']
    ITERS_PER_LR_DECAY = config['iters_per_lr_decay']
    K = config['k']
    LAMBDA = config['lambda']

    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

    train_edge_index = train_edge_index.to(device)
    val_edge_index = val_edge_index.to(device)

    early_stopping_patience = 10
    early_stopping_counter = 0
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    # pretraining
    for iter in range(ITERATIONS):
        # Forward pass
        users_emb_final, users_emb_0, items_emb_final, items_emb_0 = model.forward(
            train_sparse_edge_index,num_users,num_items)

        # Mini-batch sampling
        user_indices, pos_item_indices, neg_item_indices = sample_mini_batch( BATCH_SIZE, train_edge_index,num_items)
        user_indices, pos_item_indices, neg_item_indices = user_indices.to( device), pos_item_indices.to(device), neg_item_indices.to(device)

        # Compute embeddings for mini-batch
        users_emb_final, users_emb_0 = users_emb_final[user_indices], users_emb_0[user_indices]
        pos_items_emb_final, pos_items_emb_0 = items_emb_final[pos_item_indices], items_emb_0[pos_item_indices]
        neg_items_emb_final, neg_items_emb_0 = items_emb_final[neg_item_indices], items_emb_0[neg_item_indices]

        # Compute BPR loss
        train_loss = bpr_loss(users_emb_final, users_emb_0, pos_items_emb_final,pos_items_emb_0, neg_items_emb_final, neg_items_emb_0, LAMBDA)

        # Backpropagation and optimization
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if iter % ITERS_PER_EVAL == 0:
            model.eval()
            val_loss, recall, precision, HR,ndcg = evaluation(
                model, val_edge_index, val_sparse_edge_index, [train_edge_index], K, LAMBDA,num_users,num_items)
            # print(f"[Iteration {iter}/{ITERATIONS}] train_loss: {round(train_loss.item(), 5)}, val_loss: {round(val_loss.item(), 5)}, val_recall@{K}: {round(recall, 5)}, val_precision@{K}: {round(precision, 5)},val_HR@{K}: {round(HR, 5)}, val_ndcg@{K}: {round(ndcg, 5)}")
            train_losses.append(train_loss.item())
            val_losses.append(val_loss)

            # early stop
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {iter} iterations.")
                    break
            model.train()

        # Learning rate decay
        if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
            scheduler.step()

    # prompt-tuning
    num_users2, num_items2 = len(user_mapping2), len(item_mapping2)
    num_interactions2 = edge_index2.shape[1]
    all_indices2 = [i for i in range(num_interactions2)]

    # Split the interactions into train, validation, and test sets
    train_indices2, test_indices2 = train_test_split(all_indices2, test_size=0.2, random_state=1)
    val_indices2, test_indices2 = train_test_split(test_indices2, test_size=0.5, random_state=1)

    train_edge_index2 = edge_index2[:, train_indices2]
    val_edge_index2 = edge_index2[:, val_indices2]
    test_edge_index2 = edge_index2[:, test_indices2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Prompt-tuning - Using device {device}.")

    # define contants
    ITERATIONS =config['iterations2']
    BATCH_SIZE = config['batch_size2']
    LR = config['lr2']
    ITERS_PER_EVAL = config['iters_per_eval2']
    ITERS_PER_LR_DECAY =config['iters_per_lr_decay2']
    K = config['k2']
    p_num =config['p_num']

    # Initialize prompt
    in_channels = model.embedding_dim
    prompt = GPFplusAtt(in_channels=in_channels, p_num=p_num).to(device)
    prompt_optimizer = optim.Adam(prompt.parameters(), lr=LR)
    scheduler2 = optim.lr_scheduler.ExponentialLR(prompt_optimizer, gamma=0.95)

    model.train()
    # Load pre-trained teacher model
    teacher_model = LightGCN(num_users=num_users2, num_items=num_items2)
    teacher_model.load_state_dict(torch.load('model/mx_lightgcn.pth'))
    teacher_model.eval()
    teacher_model = teacher_model.to(device)

    # Loss weight settings
    alpha = config['alpha']
    beta = config['beta']

    train_edge_index2 = train_edge_index2.to(device)
    val_edge_index2 = val_edge_index2.to(device)

    train_losses2 = []
    val_losses2 = []

    users_emb_final2_eval=0
    items_emb_final2_eval=0

    # Training loop
    for iter in range(ITERATIONS):
        # Get embeddings from teacher model
        with torch.no_grad():
            teacher_users_emb_final, _, teacher_items_emb_final, _ = teacher_model.forward(
                train_sparse_edge_index, num_users, num_items)

        # Prompt-tuning for user and item embeddings
        users_emb_final2 = model.users_emb.weight
        items_emb_final2 = model.items_emb.weight

        users_emb_final2 = prompt.add(users_emb_final2)
        items_emb_final2 = prompt.add(items_emb_final2)
        users_emb_final2_eval =  users_emb_final2
        items_emb_final2_eval = items_emb_final2

        # Sample mini-batch for training
        user_indices2, pos_item_indices2, neg_item_indices2 = sample_mini_batch( BATCH_SIZE, train_edge_index2,num_items2)
        user_indices2, pos_item_indices2, neg_item_indices2 = user_indices2.to(device), pos_item_indices2.to(device), neg_item_indices2.to(device)

        distill_loss = distillation_loss(users_emb_final2, teacher_users_emb_final) + \
                       distillation_loss(items_emb_final2, teacher_items_emb_final)

        users_emb_final2 = users_emb_final2[user_indices2]
        pos_items_emb_final2 = items_emb_final2[pos_item_indices2]
        neg_items_emb_final2 = items_emb_final2[neg_item_indices2]

        train_loss2 = cross_entropy_loss(users_emb_final2, pos_items_emb_final2, neg_items_emb_final2)
        total_loss = alpha * distill_loss + beta * train_loss2

        # Backpropagation and optimizer step
        prompt_optimizer.zero_grad()
        total_loss.backward()
        prompt_optimizer.step()

        if iter % ITERS_PER_LR_DECAY == 0 and iter != 0:
            scheduler2.step()

        if iter % ITERS_PER_EVAL == 0:
            model.eval()
            val_loss, recall, precision,HR, ndcg = evaluation2(
                model, val_edge_index2,  [train_edge_index2], K, num_items2, users_emb_final2_eval,items_emb_final2_eval)
            print(f"[Iteration {iter}/{ITERATIONS}] train_loss: {round(train_loss2.item(), 5)}, val_loss: {round(val_loss.item(), 5)}, val_recall@{K}: {round(recall, 5)}, val_precision@{K}: {round(precision, 5)}, val_HR@{K}: {round(HR, 5)},val_ndcg@{K}: {round(ndcg, 5)}")
            train_losses2.append(train_loss2.item())
            val_losses2.append(val_loss)

            model.train()
        if iter == ITERATIONS-1:
            users_emb_final2, pos_items_emb_final2, neg_items_emb_final2, users_emb_final2_eval, items_emb_final2_eval = users_emb_final2, pos_items_emb_final2, neg_items_emb_final2,users_emb_final2_eval,items_emb_final2_eval

    #evaluation on the test set
    model.eval()
    test_edge_index2 = test_edge_index2.to(device)
    test_loss2, test_recall2, test_precision2,test_HR2, test_ndcg2 = evaluation2(
                model, test_edge_index2, [train_edge_index2], K, num_items2, users_emb_final2_eval,items_emb_final2_eval)
    print(f"[test_loss: {round(test_loss2.item(), 5)}, test_recall@{K}: {round(test_recall2, 5)}, test_precision@{K}: {round(test_precision2, 5)}, test_HR@{K}: {round(test_HR2, 5)}, test_ndcg@{K}: {round(test_ndcg2, 5)}")

if __name__ == '__main__':
    run(config)