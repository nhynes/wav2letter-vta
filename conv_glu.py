import argparse
from collections import defaultdict, OrderedDict

import nnvm.compiler
import tvm
from tvm import relay
import vta


def _conv(data, ch_out, k, s, p, name):
    weight = relay.var(f'{name}_weight')
    bias = relay.var(f'{name}_bias')
    conv = relay.nn.conv2d(data,
                           weight,
                           channels=ch_out,
                           kernel_size=(1, k),
                           strides=(1, s),
                           padding=(0, p))
    return relay.nn.bias_add(conv, bias)


def _dense(data, units, name):
    weight = relay.var(f'{name}_weight')
    bias = relay.var(f'{name}_bias')
    return relay.nn.bias_add(
        relay.nn.dense(data, weight, units), bias)


def make_net(network_arch, nfeat, nlabel, max_inp_len):
    out = relay.var('data', shape=(1, nfeat, 1, max_inp_len), dtype='float32')  # NCHW

    needs_flat = True
    itm_ctr = defaultdict(int)
    for i, line in enumerate(network_arch):
        if line.startswith('#'):
            continue

        layer = line.rstrip().split(' ')
        itm_ctr[layer[0]] += 1
        if layer[0] == 'V':  # view
            continue

        elif layer[0] == 'C':
            if layer[1] == 'NFEAT':
                layer[1] = nfeat
            _ch_in, ch_out, k, s, p = map(int, layer[1:])
            assert s == 1
            out = _conv(out, ch_out, k, s, p, name=f'conv_{itm_ctr[layer[0]]}')

        elif layer[0] == 'GLU':
            # axis = int(layer[1])
            out = relay.nn.glu(out, 1)

        elif layer[0] == 'DO':
            out = relay.nn.dropout(out, float(layer[1]))

        elif layer[0] == 'RO':  # reorder
            pass
            # out = relay.transpose(out, tuple(map(int, layer[1:])))

        elif layer[0] == 'L':  # linear
            if layer[2] == 'NLABEL':
                layer[2] = nlabel
            if needs_flat:
                out = relay.reshape(out, (1, 908))
            out = _dense(out, int(layer[2]), name=f'dense_{itm_ctr[layer[0]]}')
            needs_flat = False

        else:
            raise RuntimeError('Unknown layer type: ' + layer[0])

    args = relay.ir_pass.free_vars(out)
    return relay.Function(args, out)


def make_params(net, af_params_bin):
    import arrayfire as af
    # flashlight uses data_layout=NHCW and weight_layout=WHIO.
    # tvm prefers NCHW and OIHW
    net = relay.ir_pass.infer_type(net)
    typ_dict = OrderedDict((p.name_hint, p.checked_type) for p in net.params[1:])
    params_dict = {}
    for i, (name, typ) in enumerate(typ_dict.items()):
        tvm_shape = tuple(map(int, typ.shape))

        arr = af.array.read_array(af_params_bin, index=i).to_ndarray()
        if len(tvm_shape) == 1:
            arr = arr.flatten()
        if arr.ndim == 4:
            arr = arr.transpose(3, 2, 1, 0)

        assert arr.shape == tvm_shape, f'{arr.shape} != {tvm_shape}'

        params_dict[name] = tvm.nd.array(arr)

    return params_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network-arch',
                        type=argparse.FileType('r'), default='network.arch')
    parser.add_argument('--af-params', default='w2l_params_af.bin')
    parser.add_argument('--nfeat', type=int, default=40)
    parser.add_argument('--nlabel', type=int, default=30)
    parser.add_argument('--max-inp-len', type=int, default=741)
    # input len of librispeech: mean=741, median=577, max=3494, std=515. heavily tailed.
    parser.add_argument('--no-params', action='store_true')
    parser.add_argument('--device', choices=('vta', 'vtacpu'), default='vta')
    args = parser.parse_args()

    net = make_net(args.network_arch, args.nfeat, args.nlabel, args.max_inp_len)
    params = make_params(net, args.af_params) if not args.no_params else None

    with relay.build_config(opt_level=3):
        if args.device == 'vta':
            net = vta.graph.clean_cast(net)
            net = vta.graph.clean_conv_fuse(net)
            # net = vta.graph.pack(net, ...)  # ???
            vta.build_config().__enter__()

        graph, lib, params = relay.build(
            net, params=params,
            target=f'llvm -device={args.device}',
            target_host='llvm')

        if args.device == 'vta':
            vta.build_config().__exit__()

    lib.export_library('wav2letter2.so')
    with open('wav2letter2.json', 'w') as f_graph_json:
        f_graph_json.write(graph)
    with open('wav2letter2.params', 'wb') as f_params:
        f_params.write(nnvm.compiler.save_param_dict(params))

if __name__ == '__main__':
    main()
