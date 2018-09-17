import chainer
import chainer.functions as F
import chainer.links as L


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


class EncoderDecoderModel(chainer.Chain):
    def __init__(self, encoder, decoder, loss_fun, acc_fun):
        super(EncoderDecoderModel, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            self.decoder = decoder
        self.loss_fun = loss_fun
        self.acc_fun = acc_fun

    def __call__(self, x, condition, t):
        condition = self.encoder(condition)
        y = self.decoder(x, condition)
        loss = self.loss_fun(y, t)
        if self.acc_fun is None:
            chainer.reporter.report({'loss': loss}, self)
        else:
            chainer.reporter.report({
                'loss': loss, 'accuracy': self.acc_fun(y, t)}, self)
        return loss
