
# dataset, model, and batch size
model_order = [
    ('mnist', 'G', 256),
    ('cifar10', 'G', 256),
    ('mnist', 'F', 32),
    ('cifar10', 'F', 32),
    ('mnist', 'E', 128),
    ('cifar10', 'E', 128),
    ('mnist', 'D', 128),
    ('cifar10', 'D', 128),
    ('mnist', 'C', 128),
    ('cifar10', 'C', 128),
    ('mnist', 'B', 256),
    ('cifar10', 'B', 256),
    ('mnist', 'A', 256),
    ('cifar10', 'A', 256)
]

# Linf perturbation radius
eps_list = {
    'mnist': [0.1, 0.3],
    'cifar10': [2.0/255.0, 8.0/255.0]
}

# training mode
method_order = ['certadv']
# method_order = ['clean', 'adv', 'certadv']

# batch
batch_size_multipler = {
    'clean': 2,
    'adv': 1,
    'certadv': 1
}

# parameters for clean training
clean_params = {
    'normalized': False,
    'optimizer': 'sgd',
    'learning_rate': 0.1,
    'weight_decay': 5e-4,
    'epochs': 40,
}

# parameters for adv training
adv_params = {
    'normalized': False,
    # retrain from clean trained model
    'retrain': False,
    'optimizer': 'sgd',
    'learning_rate': 0.1,
    'weight_decay': 5e-4,
    'epochs': 40,
    # 'eps': radius,
    'eps_iter_coef': 1.0 / 50.0,
    'clip_min': 0.0,
    'clip_max': 1.0,
    'nb_iter': 100,
    'rand_init': True,
}

# parameters for certadv
certadv_params = {
    'configpath': 'crown_ibp/config/',
    'normalized': True,
    # Zico's dual training somehow always returns NaN, so I discard it and train the models by crown-ibp.
    # 'retrain': False,
    # 'normalized': True,
    # 'optimizer': 'sgd',
    # 'learning_rate': 0.1,
    # 'weight_decay': 5e-4,
    # # 'optimizer': 'adam',
    # # 'learning_rate': 0.001,
    # 'norm_type': 'l1',
    # 'epochs': 40
    'eps_iter_coef': 1.0 / 50.0,
    'clip_min': 0.0,
    'clip_max': 1.0,
    'nb_iter': 100,
    'rand_init': True,
}
