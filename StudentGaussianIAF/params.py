# parameters of training
batchsize = 1
lr = 1e-3
ema_mu = 1
trigger = (500000, 'iteration')
annealing_interval = (200000, 'iteration')
evaluate_interval = (2, 'epoch')
snapshot_interval = (10000, 'iteration')
report_interval = (100, 'iteration')

# parameters of dataset
root = '/media/hdd1/datasets/LJSpeech-1.1'
split_seed = 71

# parameters of preprocessing
sr = 22050
n_fft = 1024
hop_length = 300
n_mels = 80
top_db = 20
quantize = 2 ** 16
length = 24000
categorical_output_dim = False

# parameters of Encoder(Deconvolution network)
upsample_factors = [15, 20]

# parameters of Decoder(WaveNet)
n_loops = [1, 1, 1, 1, 1, 1]
n_layers = [10, 10, 10, 10, 10, 10]
filter_size = 3
residual_channels = 128
dilated_channels = 256
skip_channels = 128
condition_dim = n_mels
dropout_zero_rate = 0.05

# parameters of generating
apply_dropout = False

model = '/media/hdd1/chainer-ClariNet/AutoregressiveWaveNet/2018_08_02_10_36_01/snapshot_iter_500000'
