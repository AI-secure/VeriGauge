import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import sys

import datasets
import model

APPROACH_LIST = ['PGD', 'IBP', 'FastLin', 'MILP', 'PercySDP', 'ZicoDual', 'CROWN', 'CROWN-IBP', 'LPAll' 'Diffai', 'RecurJac', 'FastLip']

dataset = 'mnist'
# source = 'test'
# selector = 'small.3'
source = 'fastlin'
selector = '2.20.reg'
# source = 'cnn_cert'
# selector = '3layer_fc_20'
skip = 500
norm = '2'
radii = 0.1

def pr(rad):
    if dataset != 'mnist':
        return f'{rad*255:.3}/255'
    else:
        return f'{rad:.3}'

if __name__ == '__main__':
    ds = datasets.get_dataset(dataset, 'test')
    print(dataset)
    m = model.load_model(source, dataset, selector)
    print(m)

    from adaptor.basic_adaptor import PGDAdaptor, CWAdaptor
    from adaptor.basic_adaptor import CleanAdaptor, FastLinIBPAdaptor, MILPAdaptor, PercySDPAdaptor, FazlybSDPAdaptor
    from adaptor.lpdual_adaptor import ZicoDualAdaptor
    from adaptor.crown_adaptor import FullCrownAdaptor, CrownIBPAdaptor
    from adaptor.crown_adaptor import IBPAdaptor
    from adaptor.recurjac_adaptor import FastLipAdaptor, RecurJacAdaptor, SpectralAdaptor
    from adaptor.recurjac_adaptor import FastLinAdaptor
    from adaptor.cnncert_adaptor import CNNCertAdaptor, FastLinSparseAdaptor, LPAllAdaptor
    from adaptor.eran_adaptor import AI2Adaptor, DeepPolyAdaptor, RefineZonoAdaptor, KReluAdaptor

    cln = CleanAdaptor(dataset, m)
    pgd = PGDAdaptor(dataset, m)
    cw = CWAdaptor(dataset, m)
    # ibp = IBPAdaptor(dataset, m)
    # fastlinibp = FastLinIBPAdaptor(dataset, m)
    # milp = MILPAdaptor(dataset, m)
    sdp = PercySDPAdaptor(dataset, m)
    fazsdp = FazlybSDPAdaptor(dataset, m)
    lpdual = ZicoDualAdaptor(dataset, m)
    # fullcrown = FullCrownAdaptor(dataset, m)
    # crownibp = CrownIBPAdaptor(dataset, m)
    # fastlip = FastLipAdaptor(dataset, m)
    # recurjac = RecurJacAdaptor(dataset, m)
    # spectral = SpectralAdaptor(dataset, m)
    # fastlin = FastLinAdaptor(dataset, m)
    # cnncert = CNNCertAdaptor(dataset, m)
    # fastlinsparse = FastLinSparseAdaptor(dataset, m)
    # lpall = LPAllAdaptor(dataset, m)

    ai2 = AI2Adaptor(dataset, m)
    # deeppoly= DeepPolyAdaptor(dataset, m)
    # refinezono = RefineZonoAdaptor(dataset, m)
    # krelu = KReluAdaptorAdaptor(dataset, m)

    for i in range(0, len(ds), skip):

        X, y = ds[i]

        cln_v = cln.verify(X, y, norm, 0.0)
        # pgd_v = pgd.verify(X, y, norm, radii)
        # pgd_radius = pgd.calc_radius(X, y, norm)
        cw_v = cw.verify(X, y, norm, radii)
        cw_radius = cw.calc_radius(X, y, norm)
        # ibp_v = ibp.verify(X, y, norm, radii)
        # ibp_radius = ibp.calc_radius(X, y, norm)
        # fastlinibp_v = fastlinibp.verify(X, y, norm, radii)
        # fastlinibp_radius = fastlinibp.calc_radius(X, y, norm)
        # milp_v = milp.verify(X, y, norm, radii)
        # milp_radius = milp.calc_radius(X, y, norm, eps=1e-2)
        # sdp_v = sdp.verify(X, y, norm, radii)
        # sdp_radius = sdp.calc_radius(X, y, norm)
        # faz_v = fazsdp.verify(X, y, norm, radii)
        # faz_radius = fazsdp.calc_radius(X, y, norm)
        lpdual_v = lpdual.verify(X, y, norm, radii)
        lpdual_radius = lpdual.calc_radius(X, y, norm)

        # fullcrown_v = fullcrown.verify(X, y, norm, radii)
        # fullcrown_radius = fullcrown.calc_radius(X, y, norm)
        # crownibp_v = crownibp.verify(X, y, norm, radii)
        # crownibp_radius = crownibp.calc_radius(X, y, norm)

        # fastlip_v = fastlip.verify(X, y, norm, radii)
        # fastlip_radius = fastlip.calc_radius(X, y, norm)
        # recurjac_v = recurjac.verify(X, y, norm, radii)
        # recurjac_radius = recurjac.calc_radius(X, y, norm)
        # spectral_v = spectral.verify(X, y, norm, radii)
        # spectral_radius = spectral.calc_radius(X, y, norm)
        # fastlin_v = fastlin.verify(X, y, norm, radii)
        # fastlin_radius = fastlin.calc_radius(X, y, norm)

        # cnncert_v = cnncert.verify(X, y, norm, radii)
        # cnncert_radius = cnncert.calc_radius(X, y, norm)
        # fstlinsparse_v = fastlinsparse.verify(X, y, norm, radii)
        # fstlinsparse_radius = fastlinsparse.calc_radius(X, y, norm)
        # lpall_v = lpall.verify(X, y, norm, radii)
        # lpall_radius = lpall.calc_radius(X, y, norm)

        # ai2_v = ai2.verify(X, y, norm, radii)
        # ai2_radius = ai2.calc_radius(X, y, norm)
        # deeppoly_v = deeppoly.verify(X, y, norm, radii)
        # deeppoly_radius = deeppoly.calc_radius(X, y, norm)
        # refinezono_v = refinezono.verify(X, y, norm, radii)
        # refinezono_radius = refinezono.calc_radius(X, y, norm)
        # krelu_v = krelu.verify(X, y, norm, radii)
        # krelu_radius = krelu.calc_radius(X, y, norm)

        print(i, 'clean', cln_v,
              # 'pgd', pgd_v,
              # 'pgd_r', pr(pgd_radius),
              'cw', cw_v,
              'cw_r', pr(cw_radius),
              # 'ibp', ibp_v,
              # 'ibp_r', pr(ibp_radius),
              # 'fastlinibp', fastlinibp_v,
              # 'fastlinibp_r', pr(fastlinibp_radius),
              # 'milp', milp_v,
              # 'milp_r', pr(milp_radius),
              # 'sdp', sdp_v,
              # 'sdp_r', pr(sdp_radius),
              # 'faz', faz_v,
              # 'faz_r', pr(faz_radius),
              'lpdual', lpdual_v,
              'lpdual_r', pr(lpdual_radius),
              # 'crown', fullcrown_v,
              # 'crown_r', pr(fullcrown_radius),
              # 'crownibp', crownibp_v,
              # 'crownibp_r', pr(crownibp_radius),
              # 'fastlip', fastlip_v,
              # 'fastlip_r', pr(fastlip_radius),
              # 'recurjac', recurjac_v,
              # 'recurjac_r', pr(recurjac_radius),
              # 'spectral', spectral_v,
              # 'spectral_r', pr(spectral_radius),
              # 'fastlin', fastlin_v,
              # 'fastlin_r', pr(fastlin_radius),
              # 'cnncert', cnncert_v,
              # 'cnncert_r', pr(cnncert_radius),
              # 'fstlinsparse', fstlinsparse_v,
              # 'fstlinsparse_r', pr(fstlinsparse_radius),
              # 'lpall', lpall_v,
              # 'lpall_r', pr(lpall_radius),
              # 'ai2', ai2_v,
              # 'ai2_r', pr(ai2_radius),
              # 'deeppoly', deeppoly_v,
              # 'deeppoly_r', pr(deeppoly_radius),
              # 'refinezono', refinezono_v,
              # 'refinezono_r', pr(refinezono_radius),
              # 'krelu', krelu_v,
              # 'krelu_r', pr(krelu_radius),
              file=sys.stderr)
        # assert cln_v or not pgd_v
