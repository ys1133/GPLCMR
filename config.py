config = {}

# File path configuration
config['rating_path'] = 'data/ca_ratings.csv'
config['item_path'] = 'data/ca_items.csv'
config['rating_path2'] = 'data/mx_ratings.csv'
config['item_path2'] = 'data/mx_items.csv'

# Training parameters
config['iterations'] = 2000    # Number of training iterations
config['batch_size'] = 90      # Batch size
config['lr'] = 1e-3            # Learning rate
config['iters_per_eval'] = 100 # Iterations per evaluation
config['iters_per_lr_decay'] = 50  # Iterations per learning rate decay
config['k'] = 10               # Top K items for evaluation
config['lambda'] = 1e-6        # Regularization coefficient in loss function

# Training parameters for the second dataset
config['iterations2'] = 2000    # Number of training iterations
config['batch_size2'] = 50      # Batch size
config['lr2'] = 1e-4            # Learning rate
config['iters_per_eval2'] = 100 # Iterations per evaluation
config['iters_per_lr_decay2'] = 20  # Iterations per learning rate decay
config['k2'] = 10               # Top K items for evaluation

# Prompt learning related configuration
config['p_num'] = 5                 # Number of prompts
config['alpha'] = 0.9               # Distillation loss weight
config['beta'] = 0.9                # Standard loss weight
config['temperature'] = 3.0         # Distillation temperature
