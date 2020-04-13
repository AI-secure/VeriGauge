
import models.test_model
import models.crown_ibp_model
import models.recurjac_model
import models.cnn_cert_model
import models.exp_model


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
    },
    'cnn_cert': {
        'mnist': {
            '2layer_fc_20': lambda: models.cnn_cert_model.load_cnn_cert_model('mnist_2layer_fc_20'),
            '3layer_fc_20': lambda: models.cnn_cert_model.load_cnn_cert_model('mnist_3layer_fc_20'),
            '4layer_fc_1024': lambda: models.cnn_cert_model.load_cnn_cert_model('mnist_4layer_fc_1024'),
            'cnn_7layer': lambda: models.cnn_cert_model.load_cnn_cert_model('mnist_cnn_7layer'),
            'cnn_lenet': lambda: models.cnn_cert_model.load_cnn_cert_model('mnist_cnn_lenet'),
            'cnn_7layer_sigmoid': lambda: models.cnn_cert_model.load_cnn_cert_model('mnist_cnn_7layer_sigmoid'),
            'cnn_4layer_5_3_sigmoid': lambda: models.cnn_cert_model.load_cnn_cert_model('mnist_cnn_4layer_5_3_sigmoid'),
            'cnn_4layer_5_3_tanh': lambda: models.cnn_cert_model.load_cnn_cert_model('mnist_cnn_4layer_5_3_tanh'),
            'cnn_7layer_tanh': lambda: models.cnn_cert_model.load_cnn_cert_model('mnist_cnn_7layer_tanh'),
            'cnn_8layer_5_3_sigmoid': lambda: models.cnn_cert_model.load_cnn_cert_model('mnist_cnn_8layer_5_3_sigmoid'),
            'cnn_8layer_5_3_tanh': lambda: models.cnn_cert_model.load_cnn_cert_model('mnist_cnn_8layer_5_3_tanh'),
            'cnn_lenet_sigmoid': lambda: models.cnn_cert_model.load_cnn_cert_model('mnist_cnn_lenet_sigmoid'),
            'cnn_lenet_tanh': lambda: models.cnn_cert_model.load_cnn_cert_model('mnist_cnn_lenet_tanh'),
        },
        'cifar10': {
            '4layer_fc_2048': lambda: models.cnn_cert_model.load_cnn_cert_model('cifar_4layer_fc_2048'),
            '5layer_fc_1024': lambda: models.cnn_cert_model.load_cnn_cert_model('cifar_5layer_fc_1024'),
            '5layer_fc_2048': lambda: models.cnn_cert_model.load_cnn_cert_model('cifar_5layer_fc_2048'),
            'cnn_7layer': lambda: models.cnn_cert_model.load_cnn_cert_model('cifar_cnn_7layer'),
            '7layer_fc_1024': lambda: models.cnn_cert_model.load_cnn_cert_model('cifar_7layer_fc_1024'),
            'cnn_5layer_5_3_tanh': lambda: models.cnn_cert_model.load_cnn_cert_model('cifar_cnn_5layer_5_3_tanh'),
            'cnn_7layer_5_3_sigmoid': lambda: models.cnn_cert_model.load_cnn_cert_model('cifar_cnn_7layer_5_3_sigmoid'),
            'cnn_7layer_sigmoid': lambda: models.cnn_cert_model.load_cnn_cert_model('cifar_cnn_7layer_sigmoid'),
            'cnn_7layer_5_3_tanh': lambda: models.cnn_cert_model.load_cnn_cert_model('cifar_cnn_7layer_5_3_tanh'),
            'cnn_7layer_tanh': lambda: models.cnn_cert_model.load_cnn_cert_model('cifar_cnn_7layer_tanh'),
            'cnn_5layer_5_3_sigmoid': lambda: models.cnn_cert_model.load_cnn_cert_model('cifar_cnn_5layer_5_3_sigmoid'),
        }
    },
    'exp': {
        'mnist': {
            'A': lambda: models.exp_model.two_layer_fc20('mnist'),
            'B': lambda: models.exp_model.three_layer_fc100('mnist'),
            'C': lambda: models.exp_model.mnist_conv_small(),
            'D': lambda: models.exp_model.mnist_conv_medium(),
            'E': lambda: models.exp_model.mnist_conv_large(),
            'F': lambda: models.exp_model.conv_super('mnist'),
            'G': lambda: models.exp_model.seven_layer_fc1024('mnist')
        },
        'cifar10': {
            'A': lambda: models.exp_model.two_layer_fc20('cifar10'),
            'B': lambda: models.exp_model.three_layer_fc100('cifar10'),
            'C': lambda: models.exp_model.cifar_conv_small(),
            'D': lambda: models.exp_model.cifar_conv_medium(),
            'E': lambda: models.exp_model.cifar_conv_large(),
            'F': lambda: models.exp_model.conv_super('cifar10'),
            'G': lambda: models.exp_model.seven_layer_fc1024('cifar10')
        }
    }
}



def load_model(approach, dataset, tag='default'):
    return model_table[approach][dataset][tag]()

