import sys
import os
from os import path
import time

PATH_PREFIX = 'experiments/data'

name_mapper = {
    'A': '\\sc{FCNNa}',
    'B': '\\sc{FCNNb}',
    'G': '\\sc{FCNNc}',
    'C': '\\sc{CNNa}',
    'D': '\\sc{CNNb}',
    'E': '\\sc{CNNc}',
    'F': '\\sc{CNNd}'
}

weight_mapper = {
    'clean': 'reg',
    'adv1': 'adv1',
    'cadv1': 'cadv1',
    'adv3': 'adv3',
    'cadv3': 'cadv3',
    'adv2': 'adv2',
    'cadv2': 'cadv2',
    'adv8': 'adv8',
    'cadv8': 'cadv8'
}

name_order = "ABGCDEF"
weight_order = ['clean', 'adv1', 'adv3', 'cadv1', 'cadv3', 'adv2', 'cadv2', 'adv8', 'cadv8']
ds_weight = {
    'mnist': ('MNIST', [0, 1, 3]),
    'cifar10': ('CIFAR-10', [0, 2, 8])
}

approach_order = [
    'MILP',
    'AI2',
    'LPAll',
    'DeepPoly',
    # 'FastLin',
    'FastLinSparse',
    # 'FastLin_IBP',
    'CROWN',
    'CNNCert',
    'CROWN_IBP',
    'IBP',
    # 'IBPVer2',
    'ZicoDualLP',
    'kReLU',
    'RefineZono',
    'PercySDP',
    'FazlybSDP',
    'Spectral',
    'FastLip',
    'RecurJac',
    'PGD', 'Clean'
]

approach_mapper = {
    'MILP': 'Bounded MILP',
    'AI2': 'AI2',
    'LPAll': 'LP-Full',
    'DeepPoly': 'DeepPoly',
    # 'FastLin',
    'FastLinSparse': 'Fast-Lin',
    # 'FastLin_IBP',
    'CROWN': 'CROWN',
    'CNNCert': 'CNN-Cert',
    'CROWN_IBP': 'CROWN-IBP',
    'IBP': 'IBP',
    # 'IBPVer2',
    'ZicoDualLP': 'WK',
    'kReLU': 'k-ReLU',
    'RefineZono': 'RefineZono',
    'PercySDP': 'SDPVerify',
    'FazlybSDP': 'LMIVerify',
    'Spectral': 'Op-norm',
    'FastLip': 'FastLip',
    'RecurJac': 'RecurJac',
    'PGD': 'PGD', 'Clean': 'Clean'
}

VERIFY_TIMEOUT = 60
RADIUS_TIMEOUT = 120
TOT_SAMPLES = 100
EPS = 1e-3

def read_file(file_name, radius=False):
    with open(file_name, 'r') as f:
        contents = [x.strip().split(' ') for x in f.readlines()]
    overlap = dict()
    res = list()
    for line in contents:
        a, b, c, d, e = line
        a = int(a)
        assert b == 'True' or b == 'False'
        b = (b == 'True')
        if radius:
            c = float(c)
        else:
            c = int(c)
            assert c == 1 or c == 0
            c = bool(c)
        d = float(d)
        e = int(e)
        assert e == 1 or e == 0
        assert a not in overlap
        overlap[a] = ''
        res.append((a,b,c,d,e))
    return res


def nice_print(head_1, head_2, rows, table_1, table_2, table_3, ds, radii, tp):
    print(f'{tp} on {ds} with radii {radii}')
    for time in range(2):
        print(' '.join(['{:>13}' for _ in head_1 + [None]]).format('', *head_1))
        print(' '.join(['{:>13}' for _ in head_2 + [None]]).format('', *head_2))
        table = table_1 if time == 0 else table_2
        table = table.copy()
        for i,r in enumerate(rows):
            if time == 1:
                table[i] = ['/' if item == '' else f'{item:.2f}s/{table_3[i][j] * TOT_SAMPLES:.0f}' for (j,item) in enumerate(table[i])]
            else:
                if tp == 'verify':
                    table[i] = ['/' if item == '' else f'{item * 100.:.0f}%' for (j,item) in enumerate(table[i])]
                elif tp == 'radius':
                    table[i] = ['/' if item == '' else (f'{item:.3f}' if ds == 'mnist' else f'{item*255.:.3f}/255') for (j,item) in enumerate(table[i])]
            # table[i] = ['/' if item == '' else (f'{item * 100.:.0f}%' if time == 0 else f'{item:.2f}s/{table_3[i][j] * 100.:.0f}') for (j,item) in enumerate(table[i])]
            print(' '.join(['{:>13}' for _ in head_1 + [None]]).format(r, *(table[i])))
        print('')


def verify_texify(head_1, head_2, rows, table_1, table_2, table_3, ds, radii, handle):
    if radii == 0:
        WEIGHT_PATTERN = "\\texttt{reg}"
        RADIUS_PATTERN = "$\\epsilon=0.02$" if ds == 'mnist' else "$\\epsilon=0.5/255$"
    else:
        WEIGHT_PATTERN = "\\texttt{adv" + str(radii) + "}/\\texttt{cadv" + str(radii) + "}"
        RADIUS_PATTERN = f"$\\epsilon=0.{radii}$" if ds == 'mnist' else f"$\\epsilon={radii}/255$"
    VERIFY_CAPTION_PATTERN = f"\\emph{{Robust accuracy}} on {ds_weight[ds][0]} {WEIGHT_PATTERN} models of different verification approaches. " \
        f"The verification is on $\\cL_\\infty$ ball with {RADIUS_PATTERN} radius. " \
        f"We include results from PGD attack as the reference, which provides an upper bound."
    VERIFY_TIME_CAPTION_PATTERN = f"\\emph{{Average running time for single-instance robustness verification}} in seconds per correctly-predicted instance on {ds_weight[ds][0]} {WEIGHT_PATTERN} models of different verification approaches. " \
        "In addition, the number in the parenthesis is the timeout instances out of $100$ evaluations. "\
        f"The verification is on $\\cL_\\infty$ ball with {RADIUS_PATTERN} radius. "\
        "We stop the execution when time exceeds $\SI{60}{s}$ per instance."\
        f"We include running time of PGD attack as the reference."

    print(f'verify on {ds} with radii {radii}')

    head1repeats = 0
    while head1repeats + 1 < len(head_1) and head_1[head1repeats + 1] == head_1[head1repeats]:
        head1repeats += 1
    head1repeats += 1

    header = "{:>13} & ".format('') + " & ".join(['{:>' + str(13 * head1repeats) + '}' for _ in head_1[::head1repeats]]).format(*[f'\\{"e" if i == len(head_1[::head1repeats]) - 1 else ""}mc{{{head1repeats}}}{{{name_mapper[x]}}}' for i,x in enumerate(head_1[::head1repeats])]) + """\\\\
""" + f"\\cline{{2-{len(head_1)+1}}}\n" + \
         "{:>13} & ".format('') + " & ".join(['{:>13}' for _ in head_2]).format(*[f'\\texttt{{{x}}}' for x in head_2]) + """\\\\
"""
    verify_body_rows = [
        '{:>13}'.format(approach_mapper[r]) + ' & ' +
        ' & '.join(['{:>13}' for _ in table_1[i]])
            .format(*[f'${x * 100.:.0f}\\%$' if x != '' else '' for x in table_1[i]]) for i, r in enumerate(rows)]
    verify_body = ''
    for i, r in enumerate(rows):
        verify_body += verify_body_rows[i] + " \\\\\n"
        if i < len(rows) - 1 and (rows[i] == 'PGD' or rows[i] == 'Clean' or rows[i+1] == 'PGD' or rows[i+1] == 'Clean'):
            verify_body += '\\hline\n'

    env = "*" if len(head_1) >= 10 else ""
    width = "0.98\\textwidth" if len(head_1) >= 10 else "0.48\\textwidth"

    print(f"""
\\begin{{table{env}}}
    \\caption{{{VERIFY_CAPTION_PATTERN}}}
    \\centering
    \\resizebox{{{width}}}{{!}}{{
    \\begin{{tabular}}{{{'|'.join(['c' for _ in head_1 + [None]])}}}
    \\toprule
""" + header + "\\midrule\n" + verify_body + f"""
    \\bottomrule
    \\end{{tabular}}
    }}
    \\label{{table:exp-A-robust-accuracy-{ds}-{radii}}}
\\end{{table{env}}}
    """, file=handle)
    print(f'table:exp-A-robust-accuracy-{ds}-{radii}')

    verify_time_body_rows = [
        '{:>13}'.format(approach_mapper[r]) + ' & ' +
        ' & '.join(['{:>13}' for _ in table_2[i]])
            .format(*[f'${x:.2f}$~(${table_3[i][j] * TOT_SAMPLES:.0f}$)' if x != "" else '' for j, x in enumerate(table_2[i])]) for i, r in enumerate(rows)]
    verify_time_body = ''
    for i, r in enumerate(rows):
        verify_time_body += verify_time_body_rows[i] + " \\\\\n"
        if i < len(rows) - 1 and (rows[i] == 'PGD' or rows[i] == 'Clean' or rows[i+1] == 'PGD' or rows[i+1] == 'Clean'):
            verify_time_body += '\\hline\n'

    print(f"""
\\begin{{table{env}}}
    \\caption{{{VERIFY_TIME_CAPTION_PATTERN}}}
    \\centering
    \\resizebox{{{width}}}{{!}}{{
    \\begin{{tabular}}{{{'|'.join(['c' for _ in head_1 + [None]])}}}
    \\toprule
""" + header + "\\midrule\n" + verify_time_body + f"""
    \\bottomrule
    \\end{{tabular}}
    }}
    \\label{{table:exp-A-robust-accuracy-time-{ds}-{radii}}}
\\end{{table{env}}}
        """, file=handle)
    print(f'table:exp-A-robust-accuracy-time-{ds}-{radii}')


def radius_texify(head_1, head_2, rows, table_1, table_2, table_3, ds, radii, handle):
    if radii == 0:
        WEIGHT_PATTERN = "\\texttt{reg}"
    else:
        WEIGHT_PATTERN = "\\texttt{adv" + str(radii) + "}/\\texttt{cadv" + str(radii) + "}"
    RADIUS_CAPTION_PATTERN = f"\\emph{{Average robust radius}} on {ds_weight[ds][0]} {WEIGHT_PATTERN} models of different verification approaches. " \
        f"The verification is on $\\cL_\\infty$ ball. " \
        f"We include results from PGD attack as the reference, which provides an upper bound."
    RADIUS_TIME_CAPTION_PATTERN = f"\\emph{{Average running time for robust radius computation}} in seconds per correctly-predicted instance on {ds_weight[ds][0]} {WEIGHT_PATTERN} models of different verification approaches. " \
        "The verification is on $\\cL_\\infty$ ball. " \
        "We stop the execution when time exceeds $\SI{120}{s}$ per instance. "\
        "We include running time of PGD attack as the reference."

    # "In addition, the number in the parenthesis is the timeout instances out of $100$ evaluations. "

    print(f'radius on {ds} with radii')

    head1repeats = 0
    while head1repeats + 1 < len(head_1) and head_1[head1repeats + 1] == head_1[head1repeats]:
        head1repeats += 1
    head1repeats += 1

    header = "{:>13} & ".format('') + " & ".join(['{:>' + str(13 * head1repeats) + '}' for _ in head_1[::head1repeats]]).format(*[f'\\{"e" if i == len(head_1[::head1repeats]) - 1 else ""}mc{{{head1repeats}}}{{{name_mapper[x]}}}' for i,x in enumerate(head_1[::head1repeats])]) + """\\\\
""" + f"\\cline{{2-{len(head_1)+1}}}\n" + \
        "{:>13} & ".format('') + " & ".join(['{:>13}' for _ in head_2]).format(*[f'\\texttt{{{x}}}' for x in head_2]) + """\\\\
"""
    radius_body_rows = [
        '{:>13}'.format(approach_mapper[r]) + ' & ' +
        ' & '.join(['{:>13}' for _ in table_1[i]])
            .format(*[(f'${x:.3f}$' if ds == 'mnist' else f'${x*255.:.3f}/255$') if x != '' else '' for x in table_1[i]]) for i, r in enumerate(rows[:-1])]
    radius_body = ''
    for i, r in enumerate(rows[:-1]):
        radius_body += radius_body_rows[i] + " \\\\\n"
        if i < len(rows) - 2 and (rows[i] == 'PGD' or rows[i] == 'Clean' or rows[i+1] == 'PGD' or rows[i+1] == 'Clean'):
            radius_body += '\\hline\n'

    env = "*" if len(head_1) >= 10 else ""
    width = "0.98\\textwidth" if len(head_1) >= 10 else "0.48\\textwidth"

    print(f"""
\\begin{{table{env}}}
    \\caption{{{RADIUS_CAPTION_PATTERN}}}
    \\centering
    \\resizebox{{{width}}}{{!}}{{
    \\begin{{tabular}}{{{'|'.join(['c' for _ in head_1 + [None]])}}}
    \\toprule
""" + header + "\\midrule\n" + radius_body + f"""
    \\bottomrule
    \\end{{tabular}}
    }}
    \\label{{table:exp-A-average-radius-{ds}-{radii}}}
\\end{{table{env}}}
    """, file=handle)
    print(f'table:exp-A-average-radius-{ds}-{radii}')

    radius_time_body_rows = [
        '{:>13}'.format(approach_mapper[r]) + ' & ' +
        ' & '.join(['{:>13}' for _ in table_2[i]])
            .format(*[f'${x:.2f}$' if x != "" else '' for j, x in enumerate(table_2[i])]) for i, r in enumerate(rows[:-1])]
    # .format(*[f'${x:.2f}~$(${table_3[i][j] * TOT_SAMPLES:.0f}$)' if x != "" else '' for j, x in enumerate(table_2[i])]) for i, r in enumerate(rows)]
    radius_time_body = ''
    for i, r in enumerate(rows[:-1]):
        radius_time_body += radius_time_body_rows[i] + " \\\\\n"
        if i < len(rows) - 2 and (rows[i] == 'PGD' or rows[i] == 'Clean' or rows[i+1] == 'PGD' or rows[i+1] == 'Clean'):
            radius_time_body += '\\hline\n'

    print(f"""
\\begin{{table{env}}}
    \\caption{{{RADIUS_TIME_CAPTION_PATTERN}}}
    \\centering
    \\resizebox{{{width}}}{{!}}{{
    \\begin{{tabular}}{{{'|'.join(['c' for _ in head_1 + [None]])}}}
    \\toprule
""" + header + "\\midrule\n" + radius_time_body + f"""
    \\bottomrule
    \\end{{tabular}}
    \\label{{table:exp-A-average-radius-time-{ds}-{radii}}}
    }}
\\end{{table{env}}}
        """, file=handle)
    print(f'table:exp-A-average-radius-time-{ds}-{radii}')


if __name__ == '__main__':
    with open('experiments/tables/exp-A-robust-acc-tables.tex', 'w') as f:
        print("% This file is automatically generated by experiments/model_summary.py\n\n", file=f)

        # verify
        for ds in ds_weight:
            for radii in ds_weight[ds][1]:
                if radii == 0:
                    weights = ['clean']
                else:
                    weights = [f'adv{radii}', f'cadv{radii}']

                tab_head_1 = list()
                tab_head_2 = list()
                cur_tab_1 = list()
                cur_tab_2 = list()
                cur_tab_3 = list()
                tab_rows = list()

                records = dict()

                for iw, w in enumerate(weights):

                    for struc in name_order:

                        accs = dict()
                        avg_times = dict()
                        tle_ratios = dict()

                        # read correctness
                        file_name = f'{PATH_PREFIX}/{ds}/Clean/{struc}_{w}_verify.log'
                        # print(file_name)
                        correct_contents = read_file(file_name)
                        correct = dict()
                        for a,b,c,d,e in correct_contents:
                            # no timeout here
                            assert e == 0
                            correct[a] = b
                        assert len(correct) == TOT_SAMPLES
                        accs['Clean'] = sum(correct.keys()) / TOT_SAMPLES
                        tle_ratios['Clean'] = 0.

                        # read heuristic robustness
                        file_name = f'{PATH_PREFIX}/{ds}/PGD/{struc}_{w}_verify.log'
                        # print(file_name)
                        pgd_contents = read_file(file_name)
                        pgd_rob = dict()
                        for a,b,c,d,e in pgd_contents:
                            assert e == 0
                            assert b == correct[a]
                            pgd_rob[a] = c
                        assert len(pgd_rob) == TOT_SAMPLES
                        accs['PGD'] = sum(pgd_rob.keys()) / TOT_SAMPLES
                        avg_times['PGD'] = sum([x[3] for x in pgd_contents]) / sum([x[1] for x in pgd_contents])
                        tle_ratios['PGD'] = 0.

                        for ap in approach_order:
                            file_name = f'{PATH_PREFIX}/{ds}/{ap}/{struc}_{w}_verify.log'
                            # print(file_name)
                            if path.exists(file_name):
                                # print('exists')
                                contents = read_file(file_name)
                                for a,b,c,d,e in contents:
                                    assert b == correct[a]
                                    try:
                                        assert ap == 'PGD' or ap == 'Clean' or pgd_rob[a] == True or c == False
                                    except:
                                        print(f'MMP verify {file_name}', file=sys.stderr)
                                if len(contents) == TOT_SAMPLES:
                                    accs[ap] = sum([x[2] for x in contents]) / TOT_SAMPLES
                                    avg_times[ap] = sum([x[3] if x[4] == 0 and x[3] < VERIFY_TIMEOUT else VERIFY_TIMEOUT for x in contents]) / sum([x[1] for x in contents])
                                    tle_ratios[ap] = sum(x[4] for x in contents) / TOT_SAMPLES

                        records[(struc, w)] = (accs, avg_times, tle_ratios)

                ### to table ###
                for struc in name_order:
                    for w in weights:
                        tab_head_1.append(struc)
                        tab_head_2.append(w)
                for ap in approach_order:
                    row_1, row_2, row_3 = list(), list(), list()
                    tab_rows.append(ap)
                    for struc in name_order:
                        for w in weights:
                            if ap in records[(struc, w)][0]:
                                row_1.append(records[(struc, w)][0][ap])
                                row_2.append(records[(struc, w)][1][ap])
                                row_3.append(records[(struc, w)][2][ap])
                            else:
                                row_1.append('')
                                row_2.append('')
                                row_3.append('')
                    cur_tab_1.append(row_1)
                    cur_tab_2.append(row_2)
                    cur_tab_3.append(row_3)

                # nice_print(tab_head_1, tab_head_2, tab_rows, cur_tab_1, cur_tab_2, cur_tab_3, ds, radii, 'verify')
                verify_texify(tab_head_1, tab_head_2, tab_rows, cur_tab_1, cur_tab_2, cur_tab_3, ds, radii, f)

    with open('experiments/tables/exp-A-robust-radius-tables.tex', 'w') as f:
        print("% This file is automatically generated by experiments/model_summary.py\n\n", file=f)

        #radius
        for ds in ds_weight:
            for radii in ds_weight[ds][1]:
                if radii == 0:
                    weights = ['clean']
                else:
                    weights = [f'adv{radii}', f'cadv{radii}']

                tab_head_1 = list()
                tab_head_2 = list()
                cur_tab_1 = list()
                cur_tab_2 = list()
                cur_tab_3 = list()
                tab_rows = list()

                records = dict()

                for iw, w in enumerate(weights):

                    for struc in name_order:

                        clean_acc = None
                        rads = dict()
                        avg_times = dict()
                        tle_ratios = dict()

                        # read correctness
                        file_name = f'{PATH_PREFIX}/{ds}/Clean/{struc}_{w}_radius.log'
                        # print(file_name)
                        correct_contents = read_file(file_name, radius=True)
                        correct = dict()
                        for a, b, c, d, e in correct_contents:
                            # no timeout here
                            assert e == 0
                            correct[a] = b
                        assert len(correct) == TOT_SAMPLES
                        clean_acc = sum(correct.keys()) / TOT_SAMPLES

                        # read heuristic robustness
                        file_name = f'{PATH_PREFIX}/{ds}/PGD/{struc}_{w}_radius.log'
                        # print(file_name)
                        pgd_contents = read_file(file_name, radius=True)
                        pgd_rad = dict()
                        for a, b, c, d, e in pgd_contents:
                            assert e == 0
                            assert b == correct[a]
                            pgd_rad[a] = c
                        assert len(pgd_rad) == TOT_SAMPLES
                        rads['PGD'] = sum(pgd_rad.keys()) / sum([x[1] for x in pgd_contents])
                        avg_times['PGD'] = sum([x[3] for x in pgd_contents]) / sum([x[1] for x in pgd_contents])
                        tle_ratios['PGD'] = 0.

                        for ap in approach_order:
                            if ap == 'Clean':
                                continue
                            file_name = f'{PATH_PREFIX}/{ds}/{ap}/{struc}_{w}_radius.log'
                            # print(file_name)
                            if path.exists(file_name):
                                # print('exists')
                                contents = read_file(file_name, radius=True)
                                for a, b, c, d, e in contents:
                                    assert b == correct[a]
                                    try:
                                        assert ap == 'PGD' or ap == 'Clean' or pgd_rad[a] + EPS > c
                                    except:
                                        print(f'MMP radius {file_name}', file=sys.stderr)
                                if len(contents) == TOT_SAMPLES:
                                    rads[ap] = sum([x[2] for x in contents]) / sum([x[1] for x in pgd_contents])
                                    avg_times[ap] = sum(
                                        [x[3] if x[4] == 0 and x[3] < RADIUS_TIMEOUT else RADIUS_TIMEOUT for x in
                                         contents]) / sum([x[1] for x in contents])
                                    tle_ratios[ap] = sum(x[4] for x in contents) / TOT_SAMPLES

                        records[(struc, w)] = (rads, avg_times, tle_ratios)

                ### to table ###
                for struc in name_order:
                    for w in weights:
                        tab_head_1.append(struc)
                        tab_head_2.append(w)
                for ap in approach_order:
                    row_1, row_2, row_3 = list(), list(), list()
                    tab_rows.append(ap)
                    for struc in name_order:
                        for w in weights:
                            if ap in records[(struc, w)][0]:
                                row_1.append(records[(struc, w)][0][ap])
                                row_2.append(records[(struc, w)][1][ap])
                                row_3.append(records[(struc, w)][2][ap])
                            else:
                                row_1.append('')
                                row_2.append('')
                                row_3.append('')
                    cur_tab_1.append(row_1)
                    cur_tab_2.append(row_2)
                    cur_tab_3.append(row_3)

                # nice_print(tab_head_1, tab_head_2, tab_rows, cur_tab_1, cur_tab_2, cur_tab_3, ds, radii, 'radius')
                radius_texify(tab_head_1, tab_head_2, tab_rows, cur_tab_1, cur_tab_2, cur_tab_3, ds, radii, f)
