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
#~/anaconda3/bin/python experiments/evaluate.py \
#    --method PGD --method CROWN_IBP --method ZicoDualLP --method DeepPoly \
#    --dataset cifar10 \
#    --model A --model B --model C --model D --model E --model F --model G \
#    --mode verify --mode radius \
#    --cuda_ids 2 1>/dev/null


~/anaconda3/bin/python experiments/evaluate.py \
    --method PGD \
    --dataset cifar10 \
    --model A --model B --model C --model D --model E --model F --model G \
    --mode radius \
    --cuda_ids 2 1>/dev/null


# ====
~/anaconda3/bin/python experiments/evaluate.py \
    --method PGD \
    --dataset cifar10 \
    --model F --model G \
    --mode verify \
    --cuda_ids 2 1>/dev/null
# done


~/anaconda3/bin/python experiments/evaluate.py \
    --method CROWN_IBP --method ZicoDualLP --method DeepPoly \
    --dataset cifar10 \
    --model A --model B --model C --model D --model E --model F --model G \
    --mode verify --mode radius \
    --cuda_ids 2 1>/dev/null




~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLin_IBP --method IBPVer2 --method IBP --method Spectral \
    --dataset cifar10 \
    --model C --model D\
    --mode verify --mode radius \
    --cuda_ids 1 1>/dev/null


~/anaconda3/bin/python experiments/evaluate.py \
    --method RecurJac --method FastLip --method FastLinSparse --method FastLin \
    --dataset cifar10 \
    --model C --model D\
    --mode verify --mode radius \
    --cuda_ids 2 1>/dev/null


~/anaconda3/bin/python experiments/evaluate.py \
    --method MILP --method AI2 --method RefineZono --method LPAll \
    --dataset cifar10 \
    --model A --model B \
    --mode verify --mode radius \
    --cuda_ids 3 1>/dev/null

~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLip --weight cadv2 \
    --dataset cifar10 \
    --model E \
    --mode verify --mode radius \
    --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLip --weight cadv8 \
    --dataset cifar10 \
    --model E \
    --mode verify --mode radius \
    --cuda_ids 2 1>/dev/null;


~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLinSparse --weight clean \
    --dataset cifar10 \
    --model E \
    --mode verify --mode radius \
    --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLinSparse --weight adv2 \
    --dataset cifar10 \
    --model E \
    --mode verify --mode radius \
    --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLinSparse --weight adv8 \
    --dataset cifar10 \
    --model E \
    --mode verify \
    --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLinSparse --weight adv8 \
    --dataset cifar10 \
    --model E \
    --mode radius \
    --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLinSparse --weight cadv2 \
    --dataset cifar10 \
    --model E \
    --mode verify \
    --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLinSparse --weight cadv2 \
    --dataset cifar10 \
    --model E \
    --mode radius \
    --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLinSparse --weight cadv8 \
    --dataset cifar10 \
    --model E \
    --mode verify \
    --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLinSparse --weight cadv8 \
    --dataset cifar10 \
    --model E \
    --mode radius \
    --cuda_ids 2 1>/dev/null





~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLin --weight clean \
    --dataset cifar10 \
    --model E \
    --mode verify \
    --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLin --weight clean \
    --dataset cifar10 \
    --model E \
    --mode radius \
    --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLin --weight adv2 \
    --dataset cifar10 \
    --model E \
    --mode verify \
    --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLin --weight adv2 \
    --dataset cifar10 \
    --model E \
    --mode radius \
    --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLin --weight adv8 \
    --dataset cifar10 \
    --model E \
    --mode verify \
    --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLin --weight adv8 \
    --dataset cifar10 \
    --model E \
    --mode radius \
    --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLin --weight cadv2 \
    --dataset cifar10 \
    --model E \
    --mode verify \
    --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLin --weight cadv2 \
    --dataset cifar10 \
    --model E \
    --mode radius \
    --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLin --weight cadv8 \
    --dataset cifar10 \
    --model E \
    --mode verify \
    --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method FastLin --weight cadv8 \
    --dataset cifar10 \
    --model E \
    --mode radius \
    --cuda_ids 2 1>/dev/null


~/anaconda3/bin/python experiments/evaluate.py \
    --method PGD --method MILP --method DeepPoly --method ZicoDualLP --method CROWN --method CROWN_IBP  \
    --method CNNCert --method FastLin_IBP --method IBPVer2 --method IBP --method Spectral --method RecurJac  \
    --method FastLip --method FastLin --method FastLinSparse  \
    --weight cadv2 --weight cadv8 \
    --dataset cifar10 \
    --model B \
    --mode radius \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method PGD --method MILP --method DeepPoly --method ZicoDualLP --method CROWN --method CROWN_IBP  \
    --method CNNCert --method FastLin_IBP --method IBPVer2 --method IBP --method Spectral --method RecurJac  \
    --method FastLip --method FastLin --method FastLinSparse  \
    --weight cadv2 --weight cadv8 \
    --dataset cifar10 \
    --model A \
    --mode verify \
    --cuda_ids 2


~/anaconda3/bin/python experiments/evaluate.py \
    --method AI2  \
    --weight cadv2 \
    --dataset cifar10 \
    --model A \
    --mode verify \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method AI2  \
    --weight cadv2 \
    --dataset cifar10 \
    --model A \
    --mode radius \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method RefineZono  \
    --weight cadv2 \
    --dataset cifar10 \
    --model A \
    --mode verify \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method RefineZono  \
    --weight cadv2 \
    --dataset cifar10 \
    --model A \
    --mode radius \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method LPAll  \
    --weight cadv2 \
    --dataset cifar10 \
    --model A \
    --mode verify \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method LPAll  \
    --weight cadv2 \
    --dataset cifar10 \
    --model A \
    --mode radius \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method AI2  \
    --weight cadv2 \
    --dataset cifar10 \
    --model B \
    --mode verify \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method AI2  \
    --weight cadv2 \
    --dataset cifar10 \
    --model B \
    --mode radius \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method RefineZono  \
    --weight cadv2 \
    --dataset cifar10 \
    --model B \
    --mode verify \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method RefineZono  \
    --weight cadv2 \
    --dataset cifar10 \
    --model B \
    --mode radius \
    --cuda_ids 2 \
; and \

~/anaconda3/bin/python experiments/evaluate.py \
    --method LPAll  \
    --weight cadv2 \
    --dataset cifar10 \
    --model B \
    --mode verify \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method LPAll  \
    --weight cadv2 \
    --dataset cifar10 \
    --model B \
    --mode radius \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method AI2  \
    --weight cadv8 \
    --dataset cifar10 \
    --model A \
    --mode verify \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method AI2  \
    --weight cadv8 \
    --dataset cifar10 \
    --model A \
    --mode radius \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method RefineZono  \
    --weight cadv8 \
    --dataset cifar10 \
    --model A \
    --mode verify \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method RefineZono  \
    --weight cadv8 \
    --dataset cifar10 \
    --model A \
    --mode radius \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method LPAll  \
    --weight cadv8 \
    --dataset cifar10 \
    --model A \
    --mode verify \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method LPAll  \
    --weight cadv8 \
    --dataset cifar10 \
    --model A \
    --mode radius \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method AI2  \
    --weight cadv8 \
    --dataset cifar10 \
    --model B \
    --mode verify \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method AI2  \
    --weight cadv8 \
    --dataset cifar10 \
    --model B \
    --mode radius \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method RefineZono  \
    --weight cadv8 \
    --dataset cifar10 \
    --model B \
    --mode verify \
    --cuda_ids 2 \
; and \

~/anaconda3/bin/python experiments/evaluate.py \
    --method RefineZono  \
    --weight cadv8 \
    --dataset cifar10 \
    --model B \
    --mode radius \
    --cuda_ids 2 \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method LPAll  \
    --weight cadv8 \
    --dataset cifar10 \
    --model B \
    --mode verify \
    --cuda_ids 2 1>/dev/null \
; and \
~/anaconda3/bin/python experiments/evaluate.py \
    --method LPAll  \
    --weight cadv8 \
    --dataset cifar10 \
    --model B \
    --mode radius \
    --cuda_ids 2 1>/dev/null


~/anaconda3/bin/python experiments/evaluate.py \
    --method DeepPoly --method ZicoDualLP --method CROWN --method CROWN_IBP  \
    --method CNNCert --method FastLin_IBP --method IBPVer2 --method IBP --method Spectral --method RecurJac  \
    --method FastLip --method FastLin --method FastLinSparse  \
    --weight cadv2 --weight cadv8 \
    --dataset cifar10 \
    --model C \
    --mode verify --mode radius \
    --cuda_ids 2

~/anaconda3/bin/python experiments/evaluate.py \
                                                          --method ZicoDualLP --method CROWN  --method CROWN_IBP \
                                                          --method CNNCert --method FastLin_IBP --method IBP --method Spectral --method RecurJac  \
                                                          --method FastLip --method FastLinSparse  \
                                                          --weight cadv2 --weight cadv8 \
                                                          --dataset cifar10 \
                                                          --model G \
                                                          --mode verify --mode radius \
                                                          --cuda_ids 2 1>/dev/null

~/anaconda3/bin/python experiments/evaluate.py --method ZicoDualLP --dataset cifar10 --model E --weight cadv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method ZicoDualLP --dataset cifar10 --model E --weight cadv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method ZicoDualLP --dataset cifar10 --model E --weight cadv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method ZicoDualLP --dataset cifar10 --model E --weight cadv8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CROWN --dataset cifar10 --model E --weight cadv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CROWN --dataset cifar10 --model E --weight cadv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CROWN --dataset cifar10 --model E --weight cadv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CROWN --dataset cifar10 --model E --weight cadv8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CROWN_IBP --dataset cifar10 --model E --weight cadv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CROWN_IBP --dataset cifar10 --model E --weight cadv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CROWN_IBP --dataset cifar10 --model E --weight cadv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CROWN_IBP --dataset cifar10 --model E --weight cadv8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CNNCert --dataset cifar10 --model E --weight cadv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CNNCert --dataset cifar10 --model E --weight cadv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CNNCert --dataset cifar10 --model E --weight cadv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CNNCert --dataset cifar10 --model E --weight cadv8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastLin_IBP --dataset cifar10 --model E --weight cadv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastLin_IBP --dataset cifar10 --model E --weight cadv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastLin_IBP --dataset cifar10 --model E --weight cadv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastLin_IBP --dataset cifar10 --model E --weight cadv8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method IBP --dataset cifar10 --model E --weight cadv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method IBP --dataset cifar10 --model E --weight cadv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method IBP --dataset cifar10 --model E --weight cadv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method IBP --dataset cifar10 --model E --weight cadv8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method Spectral --dataset cifar10 --model E --weight cadv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method Spectral --dataset cifar10 --model E --weight cadv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method Spectral --dataset cifar10 --model E --weight cadv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method Spectral --dataset cifar10 --model E --weight cadv8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method RecurJac --dataset cifar10 --model E --weight cadv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method RecurJac --dataset cifar10 --model E --weight cadv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method RecurJac --dataset cifar10 --model E --weight cadv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method RecurJac --dataset cifar10 --model E --weight cadv8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastLip --dataset cifar10 --model E --weight cadv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastLip --dataset cifar10 --model E --weight cadv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastLip --dataset cifar10 --model E --weight cadv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastLip --dataset cifar10 --model E --weight cadv8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastLinSparse --dataset cifar10 --model E --weight cadv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastLinSparse --dataset cifar10 --model E --weight cadv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastLinSparse --dataset cifar10 --model E --weight cadv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastLinSparse --dataset cifar10 --model E --weight cadv8 --mode radius --cuda_ids 2 1>/dev/null

~/anaconda3/bin/python experiments/evaluate.py --method ZicoDualLP --dataset cifar10 --model F --weight cert2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method ZicoDualLP --dataset cifar10 --model F --weight cert2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method ZicoDualLP --dataset cifar10 --model F --weight cert8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method ZicoDualLP --dataset cifar10 --model F --weight cert8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CROWN --dataset cifar10 --model F --weight cert2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CROWN --dataset cifar10 --model F --weight cert2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CROWN --dataset cifar10 --model F --weight cert8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CROWN --dataset cifar10 --model F --weight cert8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CROWN_IBP --dataset cifar10 --model F --weight cert2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CROWN_IBP --dataset cifar10 --model F --weight cert2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CROWN_IBP --dataset cifar10 --model F --weight cert8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CROWN_IBP --dataset cifar10 --model F --weight cert8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CNNCert --dataset cifar10 --model F --weight cert2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CNNCert --dataset cifar10 --model F --weight cert2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CNNCert --dataset cifar10 --model F --weight cert8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method CNNCert --dataset cifar10 --model F --weight cert8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastLin_IBP --dataset cifar10 --model F --weight cert2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastLin_IBP --dataset cifar10 --model F --weight cert2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastLin_IBP --dataset cifar10 --model F --weight cert8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastLin_IBP --dataset cifar10 --model F --weight cert8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method IBP --dataset cifar10 --model F --weight cert2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method IBP --dataset cifar10 --model F --weight cert2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method IBP --dataset cifar10 --model F --weight cert8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method IBP --dataset cifar10 --model F --weight cert8 --mode radius --cuda_ids 2 1>/dev/null


~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model A --weight clean --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model A --weight clean --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model A --weight adv2 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model A --weight adv2 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model A --weight adv8 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model A --weight adv8 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model A --weight cadv2 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model A --weight cadv2 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model A --weight cadv8 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model A --weight cadv8 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model B --weight clean --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model B --weight clean --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model B --weight adv2 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model B --weight adv2 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model B --weight adv8 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model B --weight adv8 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model B --weight cadv2 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model B --weight cadv2 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model B --weight cadv8 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model B --weight cadv8 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model G --weight clean --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model G --weight clean --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model G --weight adv2 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model G --weight adv2 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model G --weight adv8 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model G --weight adv8 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model G --weight cadv2 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model G --weight cadv2 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model G --weight cadv8 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model G --weight cadv8 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model C --weight clean --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model C --weight clean --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model C --weight adv2 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model C --weight adv2 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model C --weight adv8 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model C --weight adv8 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model C --weight cadv2 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model C --weight cadv2 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model C --weight cadv8 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model C --weight cadv8 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model D --weight clean --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model D --weight clean --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model D --weight adv2 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model D --weight adv2 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model D --weight adv8 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model D --weight adv8 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model D --weight cadv2 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model D --weight cadv2 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model D --weight cadv8 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model D --weight cadv8 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model E --weight clean --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model E --weight clean --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model E --weight adv2 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model E --weight adv2 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model E --weight adv8 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model E --weight adv8 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model E --weight cadv2 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model E --weight cadv2 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model E --weight cadv8 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model E --weight cadv8 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model F --weight clean --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model F --weight clean --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model F --weight adv2 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model F --weight adv2 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model F --weight adv8 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model F --weight adv8 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model F --weight cadv2 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model F --weight cadv2 --mode radius --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model F --weight cadv8 --mode verify --cuda_ids 0 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method DeepPoly --dataset cifar10 --model F --weight cadv8 --mode radius --cuda_ids 0 1>/dev/null

~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model A --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model A --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model A --weight adv2 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model A --weight adv2 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model A --weight adv8 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model A --weight adv8 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model A --weight cadv2 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model A --weight cadv2 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model A --weight cadv8 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model A --weight cadv8 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model B --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model B --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model B --weight adv2 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model B --weight adv2 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model B --weight adv8 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model B --weight adv8 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model B --weight cadv2 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model B --weight cadv2 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model B --weight cadv8 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model B --weight cadv8 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model G --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model G --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model G --weight adv2 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model G --weight adv2 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model G --weight adv8 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model G --weight adv8 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model G --weight cadv2 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model G --weight cadv2 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model G --weight cadv8 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model G --weight cadv8 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model C --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model C --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model C --weight adv2 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model C --weight adv2 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model C --weight adv8 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model C --weight adv8 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model C --weight cadv2 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model C --weight cadv2 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model C --weight cadv8 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model C --weight cadv8 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model D --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model D --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model D --weight adv2 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model D --weight adv2 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model D --weight adv8 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model D --weight adv8 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model D --weight cadv2 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model D --weight cadv2 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model D --weight cadv8 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model D --weight cadv8 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model E --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model E --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model E --weight adv2 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model E --weight adv2 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model E --weight adv8 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model E --weight adv8 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model E --weight cadv2 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model E --weight cadv2 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model E --weight cadv8 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model E --weight cadv8 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model F --weight clean --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model F --weight clean --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model F --weight adv2 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model F --weight adv2 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model F --weight adv8 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model F --weight adv8 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model F --weight cadv2 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model F --weight cadv2 --mode radius --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model F --weight cadv8 --mode verify --cuda_ids 1 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method FastMILP --dataset cifar10 --model F --weight cadv8 --mode radius --cuda_ids 1 1>/dev/null


~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model G --weight clean --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model G --weight clean --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model G --weight adv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model G --weight adv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model G --weight adv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model G --weight adv8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model G --weight cadv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model G --weight cadv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model G --weight cadv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model G --weight cadv8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model C --weight clean --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model C --weight clean --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model C --weight adv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model C --weight adv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model C --weight adv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model C --weight adv8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model C --weight cadv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model C --weight cadv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model C --weight cadv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model C --weight cadv8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model D --weight clean --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model D --weight clean --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model D --weight adv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model D --weight adv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model D --weight adv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model D --weight adv8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model D --weight cadv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model D --weight cadv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model D --weight cadv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model D --weight cadv8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model E --weight clean --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model E --weight clean --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model E --weight adv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model E --weight adv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model E --weight adv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model E --weight adv8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model E --weight cadv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model E --weight cadv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model E --weight cadv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model E --weight cadv8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model F --weight clean --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model F --weight clean --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model F --weight adv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model F --weight adv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model F --weight adv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model F --weight adv8 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model F --weight cadv2 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model F --weight cadv2 --mode radius --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model F --weight cadv8 --mode verify --cuda_ids 2 1>/dev/null; and \
~/anaconda3/bin/python experiments/evaluate.py --method LPAll --dataset cifar10 --model F --weight cadv8 --mode radius --cuda_ids 2 1>/dev/null

~/anaconda3/bin/python experiments/evaluate.py --method PercySDP --dataset cifar10 --model A --weight clean --mode verify --verify_timeout 36000 --cuda_ids 2 ; and \
~/anaconda3/bin/python experiments/evaluate.py --method PercySDP --dataset cifar10 --model A --weight adv2 --mode verify --verify_timeout 36000 --cuda_ids 2 ; and \
~/anaconda3/bin/python experiments/evaluate.py --method PercySDP --dataset cifar10 --model A --weight adv8 --mode verify --verify_timeout 36000 --cuda_ids 2 ; and \
~/anaconda3/bin/python experiments/evaluate.py --method PercySDP --dataset cifar10 --model A --weight cadv2 --mode verify --verify_timeout 36000 --cuda_ids 2 ; and \
~/anaconda3/bin/python experiments/evaluate.py --method PercySDP --dataset cifar10 --model A --weight cadv8 --mode verify --verify_timeout 36000 --cuda_ids 2

~/anaconda3/bin/python experiments/evaluate.py --method kReLU --dataset cifar10 --model G --model C --model D --model E --model F --mode verify --mode radius --cuda_ids 0 1>/dev/null

