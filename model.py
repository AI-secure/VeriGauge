
import models.test_model
import models.crown_ibp_model
import models.recurjac_model

model_table = {
    "test": {
        'mnist': {
            'small.3': models.test_model.test_mnist,
            'tiny.1': lambda: models.test_model.test_mnist_tiny('models_weights/mnist-tiny-eps-1.pth'),
            'tiny.3': models.test_model.test_mnist_tiny
        },
        'cifar10': {
            'small.8': models.test_model.test_cifar10,
            'small.2': lambda: models.test_model.test_cifar10('models_weights/cifar-small-eps-2.pth'),
            'tiny.8': models.test_model.test_cifar10_tiny
        }
    },
    "crownibp": {
        'mnist': {
            'small.1': lambda: models.crown_ibp_model.crown_ibp_mnist_cnn_2layer_width_2('1'),
            'small.2': lambda: models.crown_ibp_model.crown_ibp_mnist_cnn_2layer_width_2('2'),
            'small.3': lambda: models.crown_ibp_model.crown_ibp_mnist_cnn_2layer_width_2('3'),
            'small.4': lambda: models.crown_ibp_model.crown_ibp_mnist_cnn_2layer_width_2('4'),
            'large.1': lambda: models.crown_ibp_model.crown_ibp_mnist_cnn_3layer_fixed_kernel_5_width_16('1'),
            'large.2': lambda: models.crown_ibp_model.crown_ibp_mnist_cnn_3layer_fixed_kernel_5_width_16('2'),
            'large.3': lambda: models.crown_ibp_model.crown_ibp_mnist_cnn_3layer_fixed_kernel_5_width_16('3'),
            'large.4': lambda: models.crown_ibp_model.crown_ibp_mnist_cnn_3layer_fixed_kernel_5_width_16('4'),
            'dm.1': lambda: models.crown_ibp_model.ibp_mnist_large('2'),
            'dm.2': lambda: models.crown_ibp_model.ibp_mnist_large('4'),
            'dm.3': lambda: models.crown_ibp_model.ibp_mnist_large('4'),
            'dm.4': lambda: models.crown_ibp_model.ibp_mnist_large('4'),
        },
        'cifar10': {
            'small.2': lambda: models.crown_ibp_model.crown_ibp_cifar_cnn_2layer_width_2('2'),
            'small.8': lambda: models.crown_ibp_model.crown_ibp_mnist_cnn_2layer_width_2('8'),
            'large.2': lambda: models.crown_ibp_model.crown_ibp_cifar_cnn_3layer_fixed_kernel_3_width_16('2'),
            'large.8': lambda: models.crown_ibp_model.crown_ibp_cifar_cnn_3layer_fixed_kernel_3_width_16('8'),
            'dm.2': lambda: models.crown_ibp_model.ibp_cifar_large('2'),
            'dm.8': lambda: models.crown_ibp_model.ibp_cifar_large('8'),
        }
    },
    "fastlin": {
        'mnist': {
            '2.20.reg': lambda: models.recurjac_model.abstract_load_keras_model('fastlin', 'mnist', 2, 'relu', 20, 'best'),
            '2.1024.reg': lambda: models.recurjac_model.abstract_load_keras_model('fastlin', 'mnist', 2, 'relu', 1024, 'best'),
            '3.20.reg': lambda: models.recurjac_model.abstract_load_keras_model('fastlin', 'mnist', 3, 'relu', 20, 'best'),
            '3.1024.adv': lambda: models.recurjac_model.abstract_load_keras_model('fastlin', 'mnist', 3, 'relu', 1024, 'adv_retrain'),
            '4.1024.reg': lambda: models.recurjac_model.abstract_load_keras_model('fastlin', 'mnist', 4, 'relu', 1024, 'best'),
            '4.1024.adv': lambda: models.recurjac_model.abstract_load_keras_model('fastlin', 'mnist', 4, 'relu', 1024, 'adv_retrain'),
        },
        'cifar10': {
            '5.2048.reg': lambda: models.recurjac_model.abstract_load_keras_model('fastlin', 'cifar', 5, 'relu', 2048, 'best'),
            '6.2048.reg': lambda: models.recurjac_model.abstract_load_keras_model('fastlin', 'cifar', 6, 'relu', 2048, 'best'),
            '7.1024.reg': lambda: models.recurjac_model.abstract_load_keras_model('fastlin', 'cifar', 7, 'relu', 1024, 'best'),
        }
    },
    'recurjac': {
        'mnist': {
            '2.20.reg': lambda: models.recurjac_model.abstract_load_keras_model('recurjac', 'mnist', 2, 'leaky', 20),
            '3.20.reg': lambda: models.recurjac_model.abstract_load_keras_model('recurjac', 'mnist', 3, 'leaky', 20),
            '3.1024.reg': lambda: models.recurjac_model.abstract_load_keras_model('recurjac', 'mnist', 3, 'relu', 1024),
            '3.1024.adv': lambda: models.recurjac_model.abstract_load_keras_model('recurjac', 'mnist', 3, 'relu', 1024, 'adv_retrain'),
            '4.20.reg': lambda: models.recurjac_model.abstract_load_keras_model('recurjac', 'mnist', 4, 'leaky', 20),
            '4.1024.reg': lambda: models.recurjac_model.abstract_load_keras_model('recurjac', 'mnist', 4, 'relu', 1024),
            '4.1024.adv': lambda: models.recurjac_model.abstract_load_keras_model('recurjac', 'mnist', 4, 'relu', 1024, 'adv_retrain'),
            '5.20.reg': lambda: models.recurjac_model.abstract_load_keras_model('recurjac', 'mnist', 5, 'leaky', 20),
            '5.50.reg': lambda: models.recurjac_model.abstract_load_keras_model('recurjac', 'mnist', 5, 'tanh', 50),
            '6.20.reg': lambda: models.recurjac_model.abstract_load_keras_model('recurjac', 'mnist', 6, 'leaky', 20),
            '7.20.reg': lambda: models.recurjac_model.abstract_load_keras_model('recurjac', 'mnist', 7, 'leaky', 20),
            '7.1024.reg': lambda: models.recurjac_model.abstract_load_keras_model('recurjac', 'mnist', 4, 'relu', 1024),
            '8.20.reg': lambda: models.recurjac_model.abstract_load_keras_model('recurjac', 'mnist', 8, 'leaky', 20),
            '9.20.reg': lambda: models.recurjac_model.abstract_load_keras_model('recurjac', 'mnist', 9, 'leaky', 20),
            '10.20.reg': lambda: models.recurjac_model.abstract_load_keras_model('recurjac', 'mnist', 10, 'leaky', 20),
        },
        'cifar10': {
            '10.2048.reg': lambda: models.recurjac_model.abstract_load_keras_model('recurjac', 'cifar', 10, 'relu', 2048),
        }
    }
}


def load_model(approach, dataset, tag='default'):
    return model_table[approach][dataset][tag]()

