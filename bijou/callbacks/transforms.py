from .basic_callbacks import Callback

class BatchTransformXCallback(Callback):
    _order = 2

    def __init__(self, tfm, **kwargs):
        super().__init__(**kwargs)
        self.tfm = tfm

    def begin_batch(self):
        self.learner.xb = self.tfm(self.xb)
