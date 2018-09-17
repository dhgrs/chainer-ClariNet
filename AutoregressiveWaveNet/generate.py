import argparse

import numpy
import librosa
import chainer
import tqdm

from WaveNet import WaveNet
from net import UpsampleNet
from utils import Preprocess
import params

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help='input file')
parser.add_argument('--output', '-o', default='result.wav', help='output file')
parser.add_argument('--model', '-m', help='snapshot of trained model')
parser.add_argument('--gpu', '-g', type=int, default=-1,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
if args.gpu != [-1]:
    chainer.cuda.set_max_workspace_size(2 * 512 * 1024 * 1024)
    chainer.global_config.autotune = True

# set data
path = args.input

# preprocess
n = 1  # batchsize; now suporrts only 1
inputs = Preprocess(
    params.sr, params.n_fft, params.hop_length, params.n_mels, params.top_db,
    None, params.categorical_output_dim)(path)

_, condition, _ = inputs
if params.categorical_output_dim is False or params.categorical_output_dim is None:
    input_dim = 1
else:
    input_dim = categorical_output_dim
x = numpy.zeros([n, input_dim, 1, 1], dtype=numpy.float32)
condition = numpy.expand_dims(condition, axis=0)

# make model
encoder = UpsampleNet(params.upsample_factors)
decoder = WaveNet(
    params.n_loop, params.n_layer, params.filter_size,
    params.residual_channels, params.dilated_channels, params.skip_channels,
    params.output_dim, params.quantize, params.log_scale_min,
    params.condition_dim, params.dropout_zero_rate)

# load trained parameter
chainer.serializers.load_npz(
    args.model, encoder, 'updater/model:main/encoder/')
chainer.serializers.load_npz(
    args.model, decoder, 'updater/model:main/decoder/')

if args.gpu >= 0:
    use_gpu = True
    chainer.cuda.get_device_from_id(args.gpu).use()
else:
    use_gpu = False

# forward
if use_gpu:
    x = chainer.cuda.to_gpu(x, device=args.gpu)
    condition = chainer.cuda.to_gpu(condition, device=args.gpu)
    encoder.to_gpu(device=args.gpu)
    decoder.to_gpu(device=args.gpu)
x = chainer.Variable(x)
condition = chainer.Variable(condition)
condition = encoder(condition)
decoder.initialize(n)
output = decoder.xp.zeros(condition.shape[2])

for i in range(len(output)):
    with chainer.using_config('enable_backprop', False):
        with chainer.using_config('train', params.apply_dropout):
            out = decoder.generate(x, condition[:, :, i:i + 1]).array
    if params.distribution_type == 'softmax':
        value = chainer.utils.WalkerAlias(
            chainer.functions.softmax(out).array[0, :, 0, 0])
        zeros = decoder.xp.zeros_like(x.array)
        zeros[:, value, :, :] = 1
        x = chainer.Variable(zeros)
    else:
        nr_mix = out.shape[1] // 3

        logit_probs = out[:, :nr_mix]
        means = out[:, nr_mix:2 * nr_mix]
        log_scales = out[:, 2 * nr_mix:3 * nr_mix]
        print(log_scales)
        log_scales = decoder.xp.maximum(log_scales, params.log_scale_min)

        if params.distribution_type == 'gaussian':
            distribution = chainer.distributions.Normal(
                means, log_scale=log_scales)
        elif params.distribution_type == 'logistic':
            distribution = chainer.distributions.Logistic(
                means, log_scale=log_scales)

        rand = distribution.sample().array
        if nr_mix == 1:
            rand = rand[0, 0, 0, 0]
        else:
            weights = chainer.functions.softmax(logit_probs).array[0]
            axis = chainer.utils.WalkerAlias(weights)
            rand = rand[0, axis, 0, 0]

        value = decoder.xp.squeeze(rand.astype(decoder.xp.float32))
        value /= 127.5
        value = decoder.xp.clip(value, -1, 1)
        x.array[:] = value
    output[i] = value

if use_gpu:
    output = chainer.cuda.to_cpu(output)
librosa.output.write_wav(args.output, output, params.sr)
