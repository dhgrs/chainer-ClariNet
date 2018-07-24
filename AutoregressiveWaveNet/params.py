# parameters of training
batchsize = 8
lr = 1e-3
ema_mu = 1
trigger = (500000, 'iteration')
annealing_interval = (200000, 'iteration')
evaluate_interval = (2, 'epoch')
snapshot_interval = (10000, 'iteration')
report_interval = (100, 'iteration')

# parameters of dataset
root = '/media/hdd1/datasets/VCTK-Corpus/'
split_seed = 71

# parameters of preprocessing
sr = 24000
n_fft = 1024
hop_length = 300
n_mels = 80
top_db = 20
quantize = 2 ** 16
length = 12000
categorical_output_dim = False

# parameters of Encoder(Deconvolution network)
upsample_factors = [15, 20]

# parameters of Decoder(WaveNet)
n_loop = 2
n_layer = 10
filter_size = 2
residual_channels = 128
dilated_channels = 256
skip_channels = 128
# quantize = quantize
# use_logistic = use_logistic
distribution_type = ['gaussian', 'logistic', 'softmax'][0]
n_mixture = 1
output_dim = 3 * n_mixture
log_scale_min = -7
condition_dim = n_mels
dropout_zero_rate = 0.05

# parameters of generating
use_ema = True
apply_dropout = False
