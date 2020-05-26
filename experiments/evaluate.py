import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sys import stderr
import sys
sys.path.append('.')
sys.path.append('..')
import argparse
import threading
import multiprocessing
import time
import getpass
import ctypes
import signal
import inspect

import datasets
import model
from models.exp_model import try_load_weight
from constants import METHOD_LIST

import torch

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

path_prefix = 'experiments/data'
norm_type = 'inf'
time_step = 0.05
EPS = 1e-6

def verify_wrapper(verifier, X, y, eps, stat_dict):
    pstime = time.time()
    ans = verifier.verify(X, y, norm_type, eps)
    pttime = time.time()
    stat_dict['result'] = int(ans)
    stat_dict['time'] = pttime - pstime


def radius_wrapper(verifier, X, y, stat_dict, ds_name):
    if ds_name == 'mnist':
        precision = 1e-2
    else:
        precision = 1e-3
    # if verifier.__class__.__name__ in ['FastLipAdaptor', 'RecurJacAdaptor', 'SpectralAdaptor', 'FastLinAdaptor']:
    if verifier.__class__.__name__ in ['SpectralAdaptor']:
        pstime = time.time()
        ans = verifier.calc_radius(X, y, norm_type)
        pttime = time.time()
        stat_dict['result'] = ans
        stat_dict['time'] = pttime - pstime
    elif verifier.__class__.__name__ in ['PGD', 'CW']:
        # compute upper bound
        # reimplement a binary search here
        pstime = time.time()
        l = 0.0
        r = 0.35
        stat_dict['result'] = r
        # for CIFAR we need higher precision
        while r - l > precision:
            # print(l, r)
            mid = (l + r) / 2.0
            if verifier.verify(X, y, norm_type, mid):
                l = mid
                # changed!
                stat_dict['result'] = l
                stat_dict['time'] = time.time() - pstime
            else:
                r = mid
                # changed!
                # stat_dict['result'] = r
                stat_dict['time'] = time.time() - pstime
    else:
        # reimplement a binary search here
        pstime = time.time()
        l = 0.0
        r = 0.5
        while r-l > precision:
            # print(l, r)
            mid = (l + r) / 2.0
            if verifier.verify(X, y, norm_type, mid):
                l = mid
                stat_dict['result'] = l
                stat_dict['time'] = time.time() - pstime
            else:
                r = mid
                stat_dict['time'] = time.time() - pstime

def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def terminate(thread):
    # os.killpg(thread.ident, signal.SIGTERM)
    _async_raise(thread.ident, SystemExit)
    _async_raise(thread.ident, KeyboardInterrupt)

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, action='append', choices=METHOD_LIST)
parser.add_argument('--dataset', type=str, action='append', choices=['mnist', 'cifar10'])
parser.add_argument('--model', type=str, action='append', choices=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
parser.add_argument('--mode', type=str, action='append', choices=['verify', 'radius'])
parser.add_argument('--weight', type=str, action='append')
parser.add_argument('--cuda_ids', type=str, default='0')
parser.add_argument('--samples', default=100, type=int)
parser.add_argument('--start', default=0, type=int)
parser.add_argument('--end', default=9999999, type=int)
parser.add_argument('--verify_timeout', default=60, type=int)
parser.add_argument('--radius_timeout', default=120, type=int)
parser.add_argument('--tleskip', default=10, type=int)
if __name__ == '__main__':
    # ctx = multiprocessing.get_context("spawn")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_ids

    # ====================== The supported adaptors ======================

    from adaptor.basic_adaptor import PGDAdaptor, CWAdaptor
    from adaptor.basic_adaptor import CleanAdaptor, FastLinIBPAdaptor, IBPAdaptor, MILPAdaptor, FastMILPAdaptor, PercySDPAdaptor, \
        FazlybSDPAdaptor
    from adaptor.lpdual_adaptor import ZicoDualAdaptor
    from adaptor.crown_adaptor import FullCrownAdaptor, CrownIBPAdaptor
    from adaptor.crown_adaptor import IBPAdaptor as IBPAdaptorV2
    from adaptor.recurjac_adaptor import FastLipAdaptor, RecurJacAdaptor, SpectralAdaptor
    from adaptor.recurjac_adaptor import FastLinAdaptor
    from adaptor.cnncert_adaptor import CNNCertAdaptor, FastLinSparseAdaptor, LPAllAdaptor
    from adaptor.eran_adaptor import AI2Adaptor, DeepPolyAdaptor, RefineZonoAdaptor, KReluAdaptor
    import tensorflow.keras.backend as K

    class_mapper = {
        'Clean': CleanAdaptor,
        'PGD': PGDAdaptor,
        'CW': CWAdaptor,
        'MILP': MILPAdaptor,
        'FastMILP': FastMILPAdaptor,
        'PercySDP': PercySDPAdaptor,
        'FazlybSDP': FazlybSDPAdaptor,
        'AI2': AI2Adaptor,
        'RefineZono': RefineZonoAdaptor,
        'LPAll': LPAllAdaptor,
        'kReLU': KReluAdaptor,
        'DeepPoly': DeepPolyAdaptor,
        'ZicoDualLP': ZicoDualAdaptor,
        'CROWN': FullCrownAdaptor,
        'CROWN_IBP': CrownIBPAdaptor,
        'CNNCert': CNNCertAdaptor,
        'FastLin_IBP': FastLinIBPAdaptor,
        'FastLin': FastLinAdaptor,
        'FastLinSparse': FastLinSparseAdaptor,
        'FastLip': FastLipAdaptor,
        'RecurJac': RecurJacAdaptor,
        'Spectral': SpectralAdaptor,
        'IBP': IBPAdaptor,
        'IBPVer2': IBPAdaptorV2
    }

    # ============================================

    skip_cnt = 0

    for now_ds_name in args.dataset:
        ds = datasets.get_dataset(now_ds_name, 'test')
        print('Current dataset:', now_ds_name, file=stderr)

        step = len(ds) // args.samples

        for now_method in args.method:
            print('Current Method:', now_method, file=stderr)

            for now_model in args.model:

                for now_weight_showname, now_weight_filename, now_eps in weights[now_ds_name]:
                    if args.weight is not None and now_weight_showname not in args.weight:
                        continue

                    if now_method == 'Clean':
                        now_eps = 0.0

                    skip_cnt = 0

                    for now_mode in args.mode:

                        if now_mode == 'verify':
                            skip_cnt = 0

                        print(f'Dataset={now_ds_name} Method={now_method} Model={now_model} Weight={now_weight_showname} Mode={now_mode}', file=stderr)

                        dir_path = f'{path_prefix}/{now_ds_name}/{now_method}'
                        filename = f'{now_model}_{now_weight_showname}_{now_mode}.log'

                        if not os.path.exists(dir_path):
                            os.makedirs(dir_path)
                        if os.path.exists(os.path.join(dir_path, filename)):
                            choice = getpass.getpass(
                                f'The result file {os.path.join(dir_path, filename)} already exists. Do you want to re-evaluate it? (Y(es)/R(ewrite)/others) ')
                            if choice == 'Y' or choice == 'y':
                                cond = 2
                                print('Re-evaluate selected.', file=stderr)
                            elif choice == 'R' or choice == 'r':
                                cond = 1
                                print('Re-evaluate and rewrite selected.', file=stderr)
                            else:
                                cond = 0
                                print('Skip selected.', file=stderr)
                        else:
                            cond = 2
                        if cond == 0:
                            continue

                        m = model.load_model('exp', now_ds_name, now_model)
                        m = m.cuda()
                        m, ok = try_load_weight(m, f'{now_ds_name}_{now_model}_{now_weight_filename}_best')
                        assert ok

                        clean_verifier = CleanAdaptor(now_ds_name, m)
                        verifier = class_mapper[now_method](now_ds_name, m)

                        if cond == 2:
                            handle = open(os.path.join(dir_path, filename), 'a')
                        elif cond == 1:
                            handle = open(os.path.join(dir_path, filename), 'w')

                        for i in range(0, len(ds), step):
                            if i < args.start:
                                continue
                            if i > args.end:
                                break
                            X, y = ds[i]

                            correct = clean_verifier.verify(X, y, norm_type, 0.)
                            nowtime = 0.

                            robust = 0
                            radius = 0.
                            tle = 0

                            if correct:
                                if skip_cnt > args.tleskip:
                                    # just skip, since it always TLE
                                    tle = 1
                                    if now_mode == 'verify':
                                        nowtime = args.verify_timeout + time_step
                                    elif now_mode == 'radius':
                                        nowtime = args.radius_timeout + time_step
                                else:
                                    if now_mode == 'verify':
                                        stat_dict = {'time': args.verify_timeout + time_step, 'result': 0}
                                        verify_proc = threading.Thread(target=verify_wrapper, args=(verifier, X, y, now_eps, stat_dict))
                                        stime = time.time()
                                        verify_proc.start()
                                        while time.time() - stime <= args.verify_timeout + time_step:
                                            if not verify_proc.is_alive():
                                                break
                                            time.sleep(time_step)
                                        if time.time() - stime > args.verify_timeout + time_step:
                                            if verify_proc.is_alive():
                                                terminate(verify_proc)
                                            # have to wait if it is refinezono
                                            if now_method in ['RefineZono']:
                                                while verify_proc.is_alive():
                                                    time.sleep(time_step)
                                            tle = 1
                                            nowtime = time.time() - stime
                                            robust = 0

                                            skip_cnt += 1
                                        else:
                                            nowtime = stat_dict['time']
                                            robust = stat_dict['result']

                                            if nowtime > args.verify_timeout:
                                                tle = 1
                                                skip_cnt += 1
                                            else:
                                                tle = 0
                                                skip_cnt = 0

                                    elif now_mode == 'radius':
                                        stat_dict = {'time': args.radius_timeout + time_step, 'result': 0.}
                                        radius_proc = threading.Thread(target=radius_wrapper, args=(verifier, X, y, stat_dict, now_ds_name))
                                        stime = time.time()
                                        radius_proc.start()
                                        while time.time() - stime <= args.radius_timeout + time_step:
                                            if not radius_proc.is_alive():
                                                break
                                            time.sleep(time_step)
                                        if time.time() - stime > args.radius_timeout + time_step:
                                            if radius_proc.is_alive():
                                                terminate(radius_proc)
                                            # have to wait if it is refinezono
                                            if now_method in ['RefineZono']:
                                                while radius_proc.is_alive():
                                                    time.sleep(time_step)
                                            tle = 1
                                            nowtime = time.time() - stime
                                            radius = stat_dict['result']
                                            if radius < EPS:
                                                skip_cnt += 1
                                            else:
                                                skip_cnt = 0
                                        else:
                                            nowtime = stat_dict['time']
                                            radius = stat_dict['result']

                                            if nowtime > args.radius_timeout:
                                                tle = 1
                                                skip_cnt += 1
                                            else:
                                                tle = 0
                                                skip_cnt = 0

                            if now_mode == 'verify':
                                print(f'  #{i} {"correct" if correct == 1 else "  wrong"} '
                                      f'{"    robust" if robust == 1 else "non-robust"} '
                                      f'time={nowtime:.3f} {"TLE" if tle else ""}',
                                      file=stderr)
                                handle.write(f'{i} {correct} {robust} {nowtime} {tle}\n')
                            elif now_mode == 'radius':
                                print(f'  #{i} {"correct" if correct == 1 else "  wrong"} '
                                      f'{radius:.3f}({radius*255.:.3f}/255) '
                                      f'time={nowtime:.3f} {"TLE" if tle else ""}', file=stderr)
                                handle.write(f'{i} {correct} {radius} {nowtime} {tle}\n')

                            handle.flush()

                        if cond == 1 or cond == 2:
                            handle.close()
                            K.clear_session()

    print('Finish!', file=stderr)


