import argparse

import numpy
import librosa
import chainer

from WaveNet import ParallelWaveNet
from net import UpsampleNet
from utils import Preprocess
import params

parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', help='input file')
parser.add_argument('--output', '-o', default='result.wav', help='output file')
parser.add_argument('--model', '-m', help='snapshot of trained model')
parser.add_argument('--threshold', '-t', type=int, default=30,
                    help='threshold of generated silence part')
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
    params.sr, params.n_fft, params.hop_length, params.n_mels, None)(path)

_, condition = inputs
condition = numpy.expand_dims(condition, axis=0)

# make model
encoder = UpsampleNet(params.upsample_factors)
student = ParallelWaveNet(
    params.n_loops, params.n_layers, params.filter_size,
    params.residual_channels, params.dilated_channels, params.skip_channels,
    params.condition_dim, params.dropout_zero_rate)

# load trained parameter
chainer.serializers.load_npz(
    args.model, encoder, 'updater/model:main/encoder/')
chainer.serializers.load_npz(
    args.model, student, 'updater/model:main/student/')

if args.gpu >= 0:
    use_gpu = True
    chainer.cuda.get_device_from_id(args.gpu).use()
else:
    use_gpu = False

# forward
if use_gpu:
    condition = chainer.cuda.to_gpu(condition, device=args.gpu)
    encoder.to_gpu(device=args.gpu)
    student.to_gpu(device=args.gpu)
condition = chainer.Variable(condition)
condition = encoder(condition)

with chainer.using_config('enable_backprop', False):
    with chainer.using_config('train', params.apply_dropout):
        shape = (1, 1, condition.shape[2], 1)
        z = student.xp.random.normal(0, 1, shape).astype(student.xp.float32)
        means, _ = student(z, condition)
        output = means
        output = student.xp.squeeze(output.array)

if use_gpu:
    output = chainer.cuda.to_cpu(output)

postprocessed_output = numpy.zeros_like(output)
intervals = librosa.effects.split(output, top_db=args.threshold)
for interval in intervals:
    postprocessed_output[interval[0]:interval[1]] = \
        output[interval[0]:interval[1]]
librosa.output.write_wav(args.output, postprocessed_output, params.sr)
