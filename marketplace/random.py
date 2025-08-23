from tinygrad import Device
from tinygrad import Tensor
from tinygrad import UOp
from tinygrad.dtype import DTypeLike
from tinygrad.dtype import dtypes
from tinygrad.dtype import to_dtype
from tinygrad.helpers import all_int
from tinygrad.helpers import argfix
from tinygrad.helpers import ceildiv
from tinygrad.helpers import prod


# The original version is taking two unit32 as the key, but we want to use one uint64 as the key. To make the
# compute graph simpler, let's change it a bit to use uint64 directly
# ref: https://github.com/tinygrad/tinygrad/blob/b057a90d493664d37558eb6c5447bc5bd5c15009/tinygrad/tensor.py#L496-L500
def _threefry_random_bits(key: Tensor, counts0: Tensor, counts1: Tensor) -> Tensor:
    x = (counts1.cast(dtypes.uint64) << 32) | counts0.cast(dtypes.uint64)
    x = x._apply_uop(UOp.threefry, key._broadcast_to(x.shape))
    counts0, counts1 = (
        (x & 0xFFFFFFFF).cast(dtypes.uint32),
        ((x >> 32) & 0xFFFFFFFF).cast(dtypes.uint32),
    )
    return counts0.cat(counts1)


# we mostly follow the implementation of Tinygrad's `rand` function, but we use our own given seed value
# ref: https://github.com/tinygrad/tinygrad/blob/b057a90d493664d37558eb6c5447bc5bd5c15009/tinygrad/tensor.py#L502-L549
def rand(
    *shape,
    seed: Tensor,
    counter: Tensor = 0,
    device: str | None = None,
    dtype: DTypeLike | None = None,
    contiguous: bool = True,
) -> Tensor:
    if not dtypes.is_float(dtype := to_dtype(dtype or dtypes.default_float)):
        raise ValueError(f"rand only supports float dtypes, got {dtype}")
    if not all_int(shape := argfix(*shape)) or not all(s >= 0 for s in shape):
        raise ValueError(f"invalid input {shape=}")
    if device is not None and not isinstance(device, str):
        raise ValueError(f"rand only supports single device, got {device=}")
    device = Device.canonicalize(device)

    if seed.dtype != dtypes.uint64:
        raise ValueError("Seed dtype needs to be uint32")
    if seed.ndim != 0:
        raise ValueError("Seed must be a scalar")

    # if shape has 0, return zero tensor
    if (numel := prod(shape)) == 0:
        return Tensor.zeros(shape, device=device, dtype=dtype, **kwargs)

    # how many 4 bytes random bits sets we should generate
    num = ceildiv(numel * dtype.itemsize, 4)

    # increase counter
    counter.assign(counter + num).contiguous()
    bits_count = counter - num

    # threefry random bits
    counts0 = (
        Tensor.arange(
            ceildiv(num, 2), device=device, dtype=dtypes.uint32, requires_grad=False
        )
        + bits_count
    )
    counts1 = counts0 + ceildiv(num, 2)
    bits = _threefry_random_bits(seed, counts0, counts1)[:num]

    # bitcast to uint with same number of bits
    _, nmant = dtypes.finfo(dtype)
    uint_dtype = {
        1: dtypes.uint8,
        2: dtypes.uint16,
        4: dtypes.uint32,
        8: dtypes.uint64,
    }[dtype.itemsize]
    bits = bits.bitcast(uint_dtype)
    # only randomize the mantissa bits and set the exponent to 1
    one = Tensor.ones_like(bits, device=bits.device, dtype=dtype).bitcast(uint_dtype)
    bits = bits.rshift((dtype.itemsize * 8) - nmant).bitwise_or(one)
    # bitcast back to the original dtype and reshape
    out = bits.bitcast(dtype)[:numel].sub(1).reshape(shape)
    return out.contiguous() if contiguous else out


class RandomNumberGenerator:
    def __init__(
        self, seed: Tensor, learning_rate: Tensor, counter: Tensor | None = None
    ):
        if seed.dtype != dtypes.uint64:
            raise ValueError("Seed dtype needs to be uint64")
        self.seed = seed
        self.learning_rate = learning_rate
        self.counter = counter
        if self.counter is None:
            self.counter = Tensor.zeros(dtype=dtypes.int)

    def rand(
        self,
        *shape,
        device: str | None = None,
        dtype: DTypeLike | None = None,
        contiguous: bool = True,
    ) -> Tensor:
        return rand(
            *shape,
            seed=self.seed,
            counter=self.counter,
            device=device,
            dtype=dtype,
            contiguous=contiguous,
        )

    def uniform(
        self, *shape, low=0.0, high=1.0, dtype: DTypeLike | None = None
    ) -> Tensor:
        return ((high - low) * self.rand(*shape)).cast(
            dtype or dtypes.default_float
        ) + low

    def uniform_like(self, target: Tensor, low=0.0, high=1.0):
        return self.uniform(*target.shape, low=low, high=high, dtype=target.dtype)

    def delta(self, *shape, dtype: DTypeLike | None = None):
        return self.uniform(
            *shape, low=-self.learning_rate, high=self.learning_rate, dtype=dtype
        )

    def delta_like(self, target: Tensor):
        return self.delta(*target.shape, dtype=target.dtype)
