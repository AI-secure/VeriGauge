#!/usr/bin/env bash

# model A, B, C, D, E, F, G
# method PGD, MILP, PercySDP, FazlybSDP, AI2, RefineZono, LPAll
#        kReLU, DeepPoly, ZicoDualLP, CROWN, CROWN_IBP, CNNCert, FastLin_IBP
#        FastLin, FastLinSparse, FastLip, RecurJac, Spectral, IBP, IBPVer2


# 3rd row method; mnist; full models; verify+radius
# run on window run1, cuda 2
#~/anaconda3/bin/python experiments/evaluate.py \
#    --method IBPVer2 --method IBP --method Spectral --method RecurJac --method FastLip --method FastLinSparse --method FastLin \
#    --dataset mnist \
#    --model A --model B --model C --model D --model E --model F --model G \
#    --mode verify --mode radius \
#    --cuda_ids 2 1>/dev/null
~/anaconda3/bin/python experiments/evaluate.py \
    --method Spectral --method RecurJac --method FastLip \
    --dataset mnist \
    --model C --model D --model E --model F \
    --mode verify --mode radius \
    --cuda_ids 2 1>/dev/null
~/anaconda3/bin/python experiments/evaluate.py \
    --method RecurJac --method FastLip \
    --dataset mnist \
    --model C --model D \
    --mode verify --mode radius \
    --cuda_ids 2 1>/dev/null
~/anaconda3/bin/python experiments/evaluate.py \
    --method Spectral \
    --dataset mnist \
    --model D --model E \
    --mode verify --mode radius \
    --cuda_ids 2 1>/dev/null
~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLinSparse \
    --dataset mnist \
    --model E --model F --model G \
    --mode verify --mode radius \
    --cuda_ids 1


~/anaconda3/bin/python experiments/evaluate.py \
    --method RecurJac --method FastLip --method FastLinSparse --method FastLin \
    --dataset mnist \
    --model A --model B --model C --model D --model E --model F --model G \
    --mode verify --mode radius \
    --cuda_ids 2 1>/dev/null


# 2nd row method; mnist; full models; verify+radius
# run on window run0, cuda 1
#~/anaconda3/bin/python experiments/evaluate.py \
#    --method kReLU --method DeepPoly --method ZicoDualLP --method CROWN --method CROWN_IBP --method CNNCert --method FastLin_IBP \
#    --dataset mnist \
#    --model A --model B --model C --model D --model E --model F --model G \
#    --mode verify --mode radius \
#    --cuda_ids 1 1>/dev/null


~/anaconda3/bin/python experiments/evaluate.py \
    --method kReLU \
    --dataset mnist \
    --model G \
    --mode verify --mode radius \
    --cuda_ids 1 1>/dev/null

~/anaconda3/bin/python experiments/evaluate.py \
    --method CROWN --method CROWN_IBP --method CNNCert --method FastLin_IBP \
    --dataset mnist \
    --model A --model B --model C --model D --model E --model F --model G  \
    --mode verify --mode radius \
    --cuda_ids 0 1>/dev/null


~/anaconda3/bin/python experiments/evaluate.py \
    --method DeepPoly --method ZicoDualLP \
    --dataset mnist \
    --model A --model B --model C --model D --model E --model F --model G  \
    --mode verify --mode radius \
    --cuda_ids 3 1>/dev/null


~/anaconda3/bin/python experiments/evaluate.py \
    --method PGD --method MILP --method PercySDP --method FazlybSDP --method AI2 --method RefineZono --method LPAll \
    --dataset mnist \
    --model A  \
    --mode verify --mode radius \
    --cuda_ids 0 1>/dev/null

~/anaconda3/bin/python experiments/evaluate.py \
    --method PGD --method MILP \
    --dataset mnist \
    --model B --model C --model D --model E --model F --model G \
    --mode verify --mode radius \
    --cuda_ids 0 1>/dev/null


~/anaconda3/bin/python experiments/evaluate.py \
    --method DeepPoly --method ZicoDualLP \
    --dataset mnist \
    --model A --model B --model C --model D --model E --model F --model G  \
    --mode verify --mode radius \
    --cuda_ids 3 1>/dev/null


~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model A --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model A --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model A --weight adv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model A --weight adv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model A --weight adv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model A --weight adv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model A --weight cadv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model A --weight cadv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model A --weight cadv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model A --weight cadv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model B --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model B --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model B --weight adv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model B --weight adv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model B --weight adv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model B --weight adv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model B --weight cadv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model B --weight cadv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model B --weight cadv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model B --weight cadv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model C --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model C --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model C --weight adv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model C --weight adv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model C --weight adv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model C --weight adv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model C --weight cadv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model C --weight cadv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model C --weight cadv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model C --weight cadv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model D --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model D --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model D --weight adv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model D --weight adv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model D --weight adv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model D --weight adv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model D --weight cadv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model D --weight cadv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model D --weight cadv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model D --weight cadv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model E --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model E --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model E --weight adv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model E --weight adv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model E --weight adv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model E --weight adv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model E --weight cadv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model E --weight cadv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model E --weight cadv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model E --weight cadv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model F --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model F --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model F --weight adv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model F --weight adv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model F --weight adv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model F --weight adv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model F --weight cadv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model F --weight cadv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model F --weight cadv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model F --weight cadv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model G --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model G --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model G --weight adv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model G --weight adv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model G --weight adv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model G --weight adv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model G --weight cadv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model G --weight cadv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model G --weight cadv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method PGD --dataset mnist --model G --weight cadv3 --mode radius --cuda_ids 1 1>/dev/null

def ppr(methods, dataset, models, weights, modes, cuda_ids, hide=True):
    return "; and \\\n".join([f"~/anaconda3/bin/python experiments/evaluate.py --method {method} --dataset {dataset} --model {model} --weight {weight} --mode {mode} --cuda_ids {cuda_ids} {'1>/dev/null' if hide else ''}" for method in methods for model in models for weight in weights for mode in modes])


~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset mnist --model G --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset mnist --model G --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset mnist --model G --weight adv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset mnist --model G --weight adv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset mnist --model G --weight adv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset mnist --model G --weight adv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset mnist --model G --weight cadv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset mnist --model G --weight cadv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset mnist --model G --weight cadv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset mnist --model G --weight cadv3 --mode radius --cuda_ids 1 1>/dev/null



~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model C --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model C --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model C --weight adv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model C --weight adv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model C --weight cadv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model C --weight cadv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model C --weight adv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model C --weight adv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model C --weight cadv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model C --weight cadv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model D --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model D --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model D --weight adv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model D --weight adv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model D --weight cadv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model D --weight cadv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model D --weight adv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model D --weight adv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model D --weight cadv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model D --weight cadv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model E --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model E --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model E --weight adv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model E --weight adv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model E --weight cadv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model E --weight cadv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model E --weight adv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model E --weight adv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model E --weight cadv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model E --weight cadv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model F --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model F --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model F --weight adv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model F --weight adv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model F --weight cadv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model F --weight cadv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model F --weight adv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model F --weight adv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model F --weight cadv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model F --weight cadv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model G --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model G --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model G --weight adv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model G --weight adv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model G --weight cadv1 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model G --weight cadv1 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model G --weight adv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model G --weight adv3 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model G --weight cadv3 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method MILP --dataset mnist --model G --weight cadv3 --mode radius --cuda_ids 1 1>/dev/null

~/anaconda3/bin/python experiments/evaluate.py --method Clean --dataset mnist --dataset cifar10 \
    --model A --model B --model C --model D --model E --model F --model G --mode verify --mode radius --cuda_ids 1 1>/dev/null


~/anaconda3/bin/python experiments/evaluate.py \
    --method IBP \
    --dataset mnist \
    --model A \
    --mode verify --mode radius \
    --cuda_ids 1 1>/dev/null

~/anaconda3/bin/python experiments/evaluate.py \
    --method FastMILP \
    --dataset mnist \
    --model A --model B --model G --model C --model D --model E --model F \
    --mode verify --mode radius \
    --cuda_ids 0 1>/dev/null

~/anaconda3/bin/python experiments/evaluate.py --method PercySDP --dataset mnist --model A --weight clean --mode verify --verify_timeout 36000 --cuda_ids 0 ; and \
~/anaconda3/bin/python experiments/evaluate.py --method PercySDP --dataset mnist --model A --weight adv1 --mode verify --verify_timeout 36000 --cuda_ids 0 ; and \
~/anaconda3/bin/python experiments/evaluate.py --method PercySDP --dataset mnist --model A --weight adv3 --mode verify --verify_timeout 36000 --cuda_ids 0 ; and \
~/anaconda3/bin/python experiments/evaluate.py --method PercySDP --dataset mnist --model A --weight cadv1 --mode verify --verify_timeout 36000 --cuda_ids 0 ; and \
~/anaconda3/bin/python experiments/evaluate.py --method PercySDP --dataset mnist --model A --weight cadv3 --mode verify --verify_timeout 36000 --cuda_ids 0


~/anaconda3/bin/python experiments/evaluate.py \
    --method AI2 \
    --dataset mnist \
    --model B --model G --model C --model D --model E --model F   \
    --mode verify --mode radius \
    --cuda_ids 2 1>/dev/null
