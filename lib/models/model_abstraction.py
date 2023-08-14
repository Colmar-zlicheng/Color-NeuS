from abc import ABC, abstractmethod


class ModuleAbstract(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def on_train_finished(self):
        pass

    @abstractmethod
    def on_val_finished(self):
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        pass

    @abstractmethod
    def compute_loss(self):
        pass

    def format_metric(self, mode, **kwargs):
        return ""

    def testing_step(self, batch, batch_idx):
        pass

    def inference_step(self, batch, batch_idx):
        pass
