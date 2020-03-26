
import models.test_model
import models.crown_ibp_model

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
    }
}


def load_model(approach, dataset, tag='default'):
    return model_table[approach][dataset][tag]()
