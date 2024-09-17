import math as m
from abc import ABC, abstractmethod
from typing import Any


class BaseCNNSize(ABC):
    def __init__(self):
        self.size: list[int] = []

    @abstractmethod
    def __kernel__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def get_size(self, idx: int | None = None) -> int | list[int]:
        pass


class CNNForwardSize(BaseCNNSize):
    def __init__(
            self,
            padding: int = 1,
            dilation: int = 1,
            kernel_size: int = 3,
            stride: int = 1,
    ) -> None:
        super().__init__()

        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride

    def __kernel__(self, x: int) -> int:
        return int(
            m.floor(
                (x + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1)
                / self.stride
                + 1
            )
        )

    def __call__(self, x: int, depth: int = 1, clear: bool = True) -> int:
        assert depth > 0

        if clear:
            self.size.clear()
        self.size.append(x)

        while depth > 0:
            x = self.__kernel__(x)
            self.size.append(x)
            depth -= 1

        return x

    def get_size(self, idx: int | None = None) -> int | list[int]:
        if idx is None:
            return self.size

        assert 0 <= idx < len(self.size) - 1
        return self.size[idx + 1]


class CNNBackwardSize(BaseCNNSize):
    def __init__(
            self,
            padding: int = 1,
            dilation: int = 1,
            kernel_size: int = 3,
            stride: int = 1,
    ) -> None:
        super().__init__()

        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.stride = stride

    def __kernel__(self, x: int, target: int) -> int:
        """Get the value of "output_padding" for torch.nn.ConvTranspose2d.

        Args:
            x (int): input image size
            target (int): target image size

        Returns:
            int: the value of "output_padding"
        """
        return target - (
                (x - 1) * self.stride
                - 2 * self.padding
                + self.dilation * (self.kernel_size - 1)
                + 1
        )

    def __call__(self, x: list[int]) -> None:
        assert len(x) > 1

        self.size.clear()

        for idx in range(len(x) - 1):
            self.size.append(self.__kernel__(x[idx], x[idx + 1]))

    def get_size(self, idx: int | None = None) -> int | list[int]:
        if idx is None:
            return self.size

        assert 0 <= idx < len(self.size)
        return self.size[idx]
