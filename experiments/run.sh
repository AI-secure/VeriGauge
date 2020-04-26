#!/usr/bin/env bash

# model A, B, C, D, E, F, G
# method PGD, MILP, PercySDP, FazlybSDP, AI2, RefineZono, LPAll
#        kReLU, DeepPoly, ZicoDualLP, CROWN, CROWN_IBP, CNNCert, FastLin_IBP
#        FastLin, FastLinSparse, FastLip, RecurJac, Spectral, IBP, IBPVer2


# 3rd row method; mnist; full models; verify+radius
# run on window run1, cuda 2
~/anaconda3/bin/python experiments/evaluate.py \
    --method IBPVer2 --method IBP --method Spectral --method RecurJac --method FastLip --method FastLinSparse --method FastLin \
    --dataset mnist \
    --model A --model B --model C --model D --model E --model F --model G \
    --mode verify --mode radius \
    --cuda_ids 2 1>/dev/null


# 2nd row method; mnist; full models; verify+radius
# run on window run0, cuda 1
~/anaconda3/bin/python experiments/evaluate.py \
    --method kReLU --method DeepPoly --method ZicoDualLP --method CROWN --method CROWN_IBP --method CNNCert --method FastLin_IBP \
    --dataset mnist \
    --model A --model B --model C --model D --model E --model F --model G \
    --mode verify --mode radius \
    --cuda_ids 1 1>/dev/null