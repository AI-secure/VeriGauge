import os
import argparse
import sys
sys.path.append('.')
sys.path.append('..')
import time

import torch

from model import load_model
from datasets import get_input_shape
from basic.models import model_transform


SAVE_PATH = 'models_weights/exp_models'

weights = {
    'mnist': [
        ('clean', 'clean_0', 0.02),
        ('adv1', 'adv_0.1', 0.1),
        ('adv3', 'adv_0.3', 0.3),
        ('cadv1', 'certadv_0.1', 0.1),
        ('cadv3', 'certadv_0.3', 0.3)
    ],
    'cifar10': [
        ('clean', 'clean_0', 0.5/255.),
        ('adv2', 'adv_0.00784313725490196', 2.0/255.),
        ('adv8', 'adv_0.03137254901960784', 8.0/255.),
        ('cadv2', 'certadv_0.00784313725490196', 2.0/255.),
        ('cadv8', 'certadv_0.03137254901960784', 8.0/255.)
    ]
}


SAVE_PATH = 'models_weights/exp_models'
def try_load_weight(m, name):
    try:
        d = torch.load(f'{SAVE_PATH}/{name}.pth')
        print('acc:', d['acc'], 'robacc:', d['robacc'], 'epoch:', d['epoch'], 'normalized:', d['normalized'], 'dataset:', d['dataset'], file=sys.stderr)
        m.load_state_dict(d['state_dict'])
        from models.test_model import get_normalize_layer
        print(f'load from {name}', file=sys.stderr)
        return m, True, {'acc': d['acc'], 'robacc': d['robacc'], 'epoch': d['epoch'], 'normalized': d['normalized'], 'dataset': d['dataset']}
    except:
        print('no load', file=sys.stderr)
        return m, False, None



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, action='append', choices=['mnist', 'cifar10'])
parser.add_argument('--model', type=str, action='append', choices=['C', 'D', 'E', 'F'])
parser.add_argument('--mode', type=str, choices=['transform', 'test'], default='transform')
parser.add_argument('--cuda_ids', type=str, default='0')
if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_ids
    if args.mode == 'transform':
        for now_dataset in args.dataset:
            print(f'Now on {now_dataset}', file=sys.stderr)
            for now_model in args.model:
                print(f'  Now on {now_model}', file=sys.stderr)
                for now_w, now_w_fname, _ in weights[now_dataset]:
                    print(f'    Now on {now_w}', file=sys.stderr)

                    m = load_model('exp', now_dataset, now_model)
                    m, flag, others = try_load_weight(m, f'{now_dataset}_{now_model}_{now_w_fname}_best')
                    assert flag

                    t1 = time.time()
                    print(f'      Transforming...')
                    newm = model_transform(m, get_input_shape(now_dataset))
                    print(f'      Saving... {time.time() - t1:.3f} s')
                    torch.save({
                        'state_dict': newm.state_dict(),
                        'acc': others['acc'],
                        'robacc': others['robacc'],
                        'epoch': others['epoch'],
                        'normalized': others['normalized'],
                        'dataset': others['normalized']
                    }, f'{SAVE_PATH}/transformed_{now_dataset}_{now_model}_{now_w_fname}_best.pth')
                    print(f'      Done {time.time() - t1:.3f} s')
    if args.mode == 'test':
        raise NotImplementedError
