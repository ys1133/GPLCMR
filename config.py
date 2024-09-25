
config = {}

# 文件路径配置
config['rating_path'] = 'data/ca_ratings.csv'
config['item_path'] = 'data/ca_items.csv'
config['rating_path2'] = 'data/mx_ratings.csv'
config['item_path2'] = 'data/mx_items.csv'

# 训练参数
config['iterations'] = 2000    # 训练迭代次数
config['batch_size'] = 90      # 批次大小
config['lr'] = 1e-3            # 学习率
config['iters_per_eval'] = 100 # 每次评估的迭代次数
config['iters_per_lr_decay'] = 50  # 每次学习率衰减的迭代次数
config['k'] = 10               # 评估的前K项
config['lambda'] = 1e-6        # 损失函数中的正则化系数

# 训练参数
config['iterations2'] = 2000    # 训练迭代次数
config['batch_size2'] = 50      # 批次大小
config['lr2'] = 1e-4            # 学习率
config['iters_per_eval2'] = 100 # 每次评估的迭代次数
config['iters_per_lr_decay2'] = 20  # 每次学习率衰减的迭代次数
config['k2'] = 10               # 评估的前K项

# Prompt学习相关配置
config['p_num'] = 5                 # Prompt 数量
config['alpha'] = 0.9               # 蒸馏损失权重
config['beta'] = 0.9                # 标准损失权重
config['temperature'] = 3.0         # 蒸馏温度
