from abc import ABC, abstractmethod

class AbstractEmbedder(ABC):

    @abstractmethod
    def image_embedder(self, images, device, num_workers, batch_size):
        pass

    @abstractmethod
    def text_embedder(self, images, device, num_workers, batch_size):
        pass