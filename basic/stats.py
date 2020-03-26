

# prefix = 'auto_exp_data/mnist.toy.pgdadv.0.1-0.03-'
# prefix = 'auto_exp_data/mnist.tiny.0.3-0.1-'
suffix = '.txt'

truth = dict()

if __name__ == '__main__':
    for prefix in ['auto_exp_data/mnist.toy.pgdadv.0.1-0.03-', 'auto_exp_data/mnist.tiny.0.3-0.1-', 'auto_exp_data/mnist.tiny.clean-0.03-', 'auto_exp_data/mnist.tiny.random-0.03-']:
        print(prefix)
        with open(f'{prefix}pgd-bound{suffix}', 'r') as f:
            for line in f.readlines():
                x, y = line.split(' ')
                x, y = int(x), int(y)
                truth[x] = y
        for t in ['lp', 'ibp', 'percy SDP', 'brute SDP', 'milp']:
            safe_cnt = 0
            all_cnt = 0
            print(t)
            try:
                with open(f'{prefix}{t}-verify{suffix}', 'r') as f:
                    for line in f.readlines():
                        x, y, z = line.split(' ')
                        x, y = int(x), int(y)
                        all_cnt += 1
                        if y == 0:
                            safe_cnt += 1
                        assert x in truth
                        assert not truth[x] or y
                print(t, 'rob acc:', safe_cnt / all_cnt)
            except:
                pass
        for t in ['lp', 'ibp', 'percy SDP', 'brute SDP', 'milp']:
            abs_sum = 0.0
            abs_cnt = 0
            time_sum = 0.0
            time_cnt = 0
            try:
                with open(f'{prefix}{t}-verify-detail{suffix}', 'r') as f:
                    for line in f.readlines():
                        try:
                            x, y, z, p, q = line.split(' ')
                            x, y, z, p, q = int(x), int(y), int(z), float(p), float(q)
                            abs_sum += abs(p)
                            abs_cnt += 1
                            time_sum += q
                            time_cnt += 1
                        except:
                            pass
                print(t, 'avg opt value:', abs_sum / abs_cnt, 'avg time:', time_sum / time_cnt)
            except:
                pass
        print('-------')
