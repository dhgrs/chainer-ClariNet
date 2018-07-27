import argparse
import pathlib
import datetime
import os
import shutil


try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
import chainer
from chainer.training import extensions

from WaveNet import WaveNet
from utils import Preprocess
from utils import get_LJSpeech_paths, get_VCTK_paths
from net import UpsampleNet, EncoderDecoderModel
import params


# use CPU or GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpus', '-g', type=int, default=[-1], nargs='+',
                    help='GPU IDs (negative value indicates CPU)')
parser.add_argument('--process', '-p', type=int, default=1,
                    help='Number of parallel processes')
parser.add_argument('--prefetch', '-f', type=int, default=1,
                    help='Number of prefetch samples')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the training from snapshot')
args = parser.parse_args()
if args.gpus != [-1]:
    chainer.cuda.set_max_workspace_size(2 * 512 * 1024 * 1024)
    chainer.global_config.autotune = True

# get paths
# files, _ = get_LJSpeech_paths(params.root)
files, _ = get_VCTK_paths(params.root)

preprocess = Preprocess(
    params.sr, params.n_fft, params.hop_length, params.n_mels, params.top_db,
    params.length, params.categorical_output_dim)

dataset = chainer.datasets.TransformDataset(files, preprocess)
train, valid = chainer.datasets.split_dataset_random(
    dataset, int(len(dataset) * 0.9), params.split_seed)

# make directory of results
result = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
os.mkdir(result)
shutil.copy(__file__, os.path.join(result, __file__))
shutil.copy('utils.py', os.path.join(result, 'utils.py'))
shutil.copy('params.py', os.path.join(result, 'params.py'))
shutil.copy('generate.py', os.path.join(result, 'generate.py'))
shutil.copy('net.py', os.path.join(result, 'net.py'))
shutil.copytree('WaveNet', os.path.join(result, 'WaveNet'))

# Model
encoder = UpsampleNet(params.upsample_factors)
decoder = WaveNet(
    params.n_loop, params.n_layer, params.filter_size,
    params.residual_channels, params.dilated_channels, params.skip_channels,
    params.output_dim, params.quantize, params.log_scale_min,
    params.condition_dim, params.dropout_zero_rate)

if params.distribution_type == 'gaussian':
    loss_fun = decoder.calculate_gaussian_loss
    acc_fun = None
elif params.distribution_type == 'logistic':
    loss_fun = decoder.calculate_logistic_loss
    acc_fun = None
elif params.distribution_type == 'softmax':
    loss_fun = chainer.functions.softmax_cross_entropy
    acc_fun = chainer.functions.accuracy
model = EncoderDecoderModel(encoder, decoder, loss_fun, acc_fun)

# Optimizer
optimizer = chainer.optimizers.Adam(params.lr / len(args.gpus))
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(10))

# Iterator
if args.process * args.prefetch > 1:
    train_iter = chainer.iterators.MultiprocessIterator(
        train, params.batchsize,
        n_processes=args.process, n_prefetch=args.prefetch)
    valid_iter = chainer.iterators.MultiprocessIterator(
        valid, params.batchsize // len(args.gpus), repeat=False, shuffle=False,
        n_processes=args.process, n_prefetch=args.prefetch)
else:
    train_iter = chainer.iterators.SerialIterator(train, params.batchsize)
    valid_iter = chainer.iterators.SerialIterator(
        valid, params.batchsize // len(args.gpus), repeat=False, shuffle=False)

# Updater
if args.gpus == [-1]:
    updater = chainer.training.StandardUpdater(train_iter, optimizer)
else:
    chainer.cuda.get_device_from_id(args.gpus[0]).use()
    names = ['main'] + list(range(len(args.gpus) - 1))
    devices = {str(name): gpu for name, gpu in zip(names, args.gpus)}
    updater = chainer.training.ParallelUpdater(
        train_iter, optimizer, devices=devices)

# Trainer
trainer = chainer.training.Trainer(updater, params.trigger, out=result)

# Extensions
trainer.extend(extensions.ExponentialShift('alpha', 0.5),
               trigger=params.annealing_interval)
trainer.extend(extensions.Evaluator(valid_iter, model, device=args.gpus[0]),
               trigger=params.evaluate_interval)
trainer.extend(extensions.dump_graph('main/loss'))
trainer.extend(extensions.snapshot(), trigger=params.snapshot_interval)
trainer.extend(extensions.LogReport(trigger=params.report_interval))
trainer.extend(extensions.observe_lr(), trigger=params.report_interval)
trainer.extend(extensions.PrintReport(
    ['epoch', 'iteration', 'main/loss', 'main/accuracy',
     'validation/main/loss', 'validation/main/accuracy']),
    trigger=params.report_interval)
trainer.extend(extensions.PlotReport(
    ['main/loss', 'validation/main/loss'],
    'iteration', file_name='loss.png', trigger=params.report_interval))
trainer.extend(extensions.PlotReport(
    ['main/accuracy', 'validation/main/accuracy'],
    'iteration', file_name='accuracy.png', trigger=params.report_interval))
trainer.extend(extensions.PlotReport(
    ['lr'], 'iteration', file_name='lr.png', trigger=params.report_interval))
trainer.extend(extensions.ProgressBar(update_interval=1))

if args.resume:
    chainer.serializers.load_npz(args.resume, trainer)

# run
print('GPUs: {}'.format(*args.gpus))
print('# train: {}'.format(len(train)))
print('# valid: {}'.format(len(valid)))
print('# Minibatch-size: {}'.format(params.batchsize))
print('# {}: {}'.format(params.trigger[1], params.trigger[0]))
print('')

trainer.run()
