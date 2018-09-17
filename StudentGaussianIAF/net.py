import chainer
import chainer.functions as F
import chainer.links as L
import numpy


class UpsampleNet(chainer.ChainList):
    def __init__(self, upscale_factors):
        super(UpsampleNet, self).__init__()
        for factor in upscale_factors:
            self.add_link(L.Deconvolution2D(
                None, 1, (3, 2 * factor),
                stride=(1, factor), pad=(1, factor // 2)))

    def __call__(self, x):
        for link in self.children():
            x = F.leaky_relu(link(x), 0.4)
            if (x.shape[-1] % 2) != 0:
                x = x[:, :, :, :-1]
        return x.transpose((0, 2, 3, 1))


class DistilModel(chainer.Chain):
    def __init__(
            self, encoder, teacher, student, lmd=4,
            n_fft=2048, hop_length=300, win_length=1200):
        super(DistilModel, self).__init__()
        self.lmd = lmd
        with self.init_scope():
            self.encoder = encoder
            self.teacher = teacher
            self.student = student
            self.stft = STFT(n_fft, hop_length, win_length)

    def scalar_to_tensor(self, shapeortensor, scalar):
        if hasattr(shapeortensor, 'shape'):
            shape = shapeortensor.shape
        else:
            shape = shapeortensor
        return self.xp.full(shape, scalar, dtype=self.xp.float32)

    def __call__(self, t, condition):
        # t(timesteps): 1-T

        distribution = chainer.distributions.Normal(
            self.xp.array(0, dtype='f'), self.xp.array(1, dtype='f'))
        z = distribution.sample(t.shape)
        # z(timesteps): 1-T

        condition = self.encoder(condition)
        # condition(timesteps): 1-T

        s_means, s_scales = self.student(z, condition)
        s_clipped_scales = F.maximum(
            s_scales, self.scalar_to_tensor(s_scales, -7))
        # s_means, s_scales(timesteps): 2-(T+1)

        x = z[:, :, 1:] * F.exp(s_scales[:, :, :-1]) + s_means[:, :, :-1]
        # x(timesteps): 2-T

        with chainer.using_config('train', False):
            y = self.teacher(x, condition[:, :, 1:])
        t_means, t_scales = y[:, 1:2], y[:, 2:3]
        t_clipped_scales = F.maximum(
            t_scales, self.scalar_to_tensor(t_scales, -7))
        # t_means, t_scales(timesteps): 3-(T+1)

        s_distribution = chainer.distributions.Normal(
            s_means[:, :, 1:], log_scale=s_clipped_scales[:, :, 1:])
        t_distribution = chainer.distributions.Normal(
            t_means, log_scale=t_clipped_scales)
        # s_distribution, t_distribution(timesteps): 3-(T+1)

        kl = chainer.kl_divergence(s_distribution, t_distribution)
        kl = F.minimum(
            kl, self.scalar_to_tensor(kl, 100))
        kl = F.average(kl)

        regularization = F.mean_squared_error(
            t_scales, s_scales[:, :, 1:])

        spectrogram_frame_loss = F.mean_squared_error(
            self.stft.magnitude(t[:, :, 1:]), self.stft.magnitude(x))

        loss = kl + self.lmd * regularization + spectrogram_frame_loss
        chainer.reporter.report({
            'loss': loss, 'kl_divergence': kl,
            'regularization': regularization,
            'spectrogram_frame_loss': spectrogram_frame_loss}, self)
        return loss


class STFT(chainer.Chain):

    def __init__(self, n_fft, hop_length, win_length, window=numpy.hanning):
        super(STFT, self).__init__()
        xp = self.xp
        self.hop_length = hop_length
        self.n_bin = n_fft // 2

        # calculate weights
        weight_real = xp.cos(
            -2*xp.pi*xp.arange(n_fft).reshape((n_fft, 1)) *
            xp.arange(n_fft) / n_fft)[:n_fft//2]
        weight_imag = xp.sin(
            -2*xp.pi*xp.arange(n_fft).reshape((n_fft, 1)) *
            xp.arange(n_fft) / n_fft)[:n_fft//2]

        # calculate window
        window = window(win_length)
        if n_fft != win_length:
            pad = (n_fft - win_length) // 2
            window = numpy.pad(window, [pad, pad], 'constant')
        window = self.xp.array(window.reshape((1, 1, 1, n_fft)))

        # set persistent
        self.add_persistent(
            'weight_real',
            window * weight_real.reshape((n_fft//2, 1, 1, n_fft)))
        self.add_persistent(
            'weight_imag',
            window * weight_imag.reshape((n_fft//2, 1, 1, n_fft)))

    def __call__(self, x):
        x = x.transpose((0, 1, 3, 2))
        real = F.convolution_2d(
            x, self.weight_real, stride=(1, self.hop_length))
        imag = F.convolution_2d(
            x, self.weight_imag, stride=(1, self.hop_length))
        return real, imag

    def power(self, x):
        real, imag = self(x)
        power = real ** 2 + imag ** 2
        return power

    def magnitude(self, x):
        power = self.power(x)
        magnitude = F.sqrt(power)
        return magnitude
