from tinygrad import Device
from tinygrad import Tensor
from tinygrad.dtype import DTypeLike
from tinygrad.dtype import dtypes
from tinygrad.dtype import to_dtype
from tinygrad.helpers import all_int
from tinygrad.helpers import argfix
from tinygrad.helpers import ceildiv
from tinygrad.helpers import prod


# we mostly follow the implementation of Tinygrad's `rand` function, but we use our own given seed value
# ref: https://github.com/tinygrad/tinygrad/blob/b057a90d493664d37558eb6c5447bc5bd5c15009/tinygrad/tensor.py#L502-L549
def rand(
    *shape,
    seed: Tensor,
    base_count: int | Tensor = 0,
    device: str | None = None,
    dtype: DTypeLike | None = None,
    contiguous: bool = True,
    **kwargs,
) -> Tensor:
    if not dtypes.is_float(dtype := to_dtype(dtype or dtypes.default_float)):
        raise ValueError(f"rand only supports float dtypes, got {dtype}")
    if not all_int(shape := argfix(*shape)) or not all(s >= 0 for s in shape):
        raise ValueError(f"invalid input {shape=}")
    if device is not None and not isinstance(device, str):
        raise ValueError(f"rand only supports single device, got {device=}")
    device = Device.canonicalize(device)

    if seed.dtype != dtypes.uint32:
        raise ValueError("Seed dtype needs to be uint32")
    if seed.shape != (2,):
        raise ValueError("Seed shape needs to be (2, )")

    if not isinstance(base_count, int):
        raise ValueError("Base count needs to be an integer")

    # if shape has 0, return zero tensor
    if (numel := prod(shape)) == 0:
        return Tensor.zeros(shape, device=device, dtype=dtype, **kwargs)

    # how many 4 bytes random bits sets we should generate
    num = ceildiv(numel * dtype.itemsize, 4)

    # threefry random bits
    counts0 = (
        Tensor.arange(
            ceildiv(num, 2), device=device, dtype=dtypes.uint32, requires_grad=False
        )
        + base_count
    )
    counts1 = counts0 + ceildiv(num, 2)
    bits = Tensor._threefry_random_bits(seed, counts0, counts1)[:num]

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
    out = (
        bits.bitcast(dtype)[:numel]
        .sub(1)
        .reshape(shape)
        .requires_grad_(kwargs.get("requires_grad"))
    )
    return out.contiguous() if contiguous else out
