import typing

from tinygrad import nn
from tinygrad import Tensor


def repeat(w: Tensor, count: int) -> Tensor:
    return w.unsqueeze(0).repeat(count, *((1,) * len(w.shape))).contiguous()


class MultiModelBase:
    training: typing.ClassVar[bool] = False

    vendor_count: int

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError()


class MultiConv2d(MultiModelBase, nn.Conv2d):
    def __init__(
        self,
        vendor_count: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride=1,
        padding: int | tuple[int, ...] | str = 0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.vendor_count = vendor_count
        self.weight = repeat(self.weight, vendor_count)
        if self.bias is not None:
            self.bias = repeat(self.bias, vendor_count)

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        return x.conv2d(
            self.weight[i],
            self.bias[i] if self.bias is not None else None,
            self.groups,
            self.stride,
            self.dilation,
            self.padding,
        )


class MultiLinear(MultiModelBase, nn.Linear):
    def __init__(
        self, vendor_count: int, in_features: int, out_features: int, bias: bool = True
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.vendor_count = vendor_count
        self.weight = repeat(self.weight, vendor_count)
        if self.bias is not None:
            self.bias = repeat(self.bias, vendor_count)

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        return x.linear(
            self.weight[i].transpose(), self.bias[i] if self.bias is not None else None
        )


class MultiBatchNorm(MultiModelBase, nn.BatchNorm):
    def __init__(
        self,
        vendor_count: int,
        sz: int,
        eps: float = 1e-5,
        affine: bool = True,
        momentum: float = 0.1,
    ):
        super().__init__(
            sz=sz,
            eps=eps,
            affine=affine,
            momentum=momentum,
        )
        self.vendor_count = vendor_count
        if affine:
            self.weight = repeat(self.weight, vendor_count)
            self.bias = repeat(self.bias, vendor_count)
        if hasattr(self, "running_mean"):
            self.running_mean = repeat(self.running_mean, vendor_count)
        if hasattr(self, "running_var"):
            self.running_var = repeat(self.running_var, vendor_count)
        del self.num_batches_tracked

    def calc_stats(self, i: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        shape_mask: list[int] = [1, -1, *([1] * (x.ndim - 2))]
        if self.track_running_stats and not MultiModelBase.training:
            return self.running_mean[i], self.running_var[i].reshape(
                shape=shape_mask
            ).expand(x.shape)
        # This requires two full memory accesses to x
        # https://github.com/pytorch/pytorch/blob/c618dc13d2aa23625cb0d7ada694137532a4fa33/aten/src/ATen/native/cuda/Normalization.cuh
        # There's "online" algorithms that fix this, like https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
        batch_mean = x.mean(
            axis=(reduce_axes := tuple(x for x in range(x.ndim) if x != 1))
        )
        y = x - batch_mean.detach().reshape(shape=shape_mask)  # d(var)/d(mean) = 0
        batch_var = (y * y).mean(axis=reduce_axes)
        return batch_mean, batch_var

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        batch_mean, batch_var = self.calc_stats(i, x)
        if self.track_running_stats and MultiModelBase.training:
            self.running_mean[i].assign(
                (1 - self.momentum) * self.running_mean[i]
                + self.momentum * batch_mean.detach()
            )
            self.running_var[i].assign(
                (1 - self.momentum) * self.running_var[i]
                + self.momentum
                * x.numel()
                / (x.numel() - x.shape[1])
                * batch_var.detach()
            )
        return x.batchnorm(
            self.weight[i] if self.weight is not None else None,
            self.bias[i] if self.bias is not None else None,
            batch_mean,
            batch_var.add(self.eps).rsqrt(),
        )


class MultiModel(MultiModelBase):
    def __init__(
        self,
        layers: typing.List[MultiModelBase | typing.Callable[[Tensor], Tensor]],
        vendor_count: int | None = None,
    ):
        self.vendor_count = vendor_count
        self.layers: typing.List[MultiModelBase | typing.Callable[[Tensor], Tensor]] = (
            layers
        )
        for i, model in enumerate(self.layers):
            if not isinstance(model, MultiModelBase):
                continue
            if self.vendor_count is None:
                self.vendor_count = model.vendor_count
            else:
                if model.vendor_count != self.vendor_count:
                    raise ValueError(
                        f"Layer {i} vendor count {model.vendor_count} does not match with "
                        f"the multi model's {self.vendor_count}"
                    )
        if self.vendor_count is None:
            self.vendor_count = 1

    def __call__(self, i: Tensor, x: Tensor) -> Tensor:
        value = x
        for model in self.layers:
            if isinstance(model, MultiModelBase):
                value = model(i, value)
            else:
                value = model(value)
        return value
