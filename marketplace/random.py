import hashlib

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
    seed: Tensor,
    *shape,
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

    # if shape has 0, return zero tensor
    if (numel := prod(shape)) == 0:
        return Tensor.zeros(shape, device=device, dtype=dtype, **kwargs)
    num = ceildiv(numel * dtype.itemsize, 4)

    # generate per device seeds and rng counter if we haven't seen this device yet
    if device not in Tensor._device_seeds:
        Tensor._device_seeds[device] = Tensor(
            [
                int.from_bytes(
                    hashlib.sha256(
                        len(Tensor._device_seeds).to_bytes(4, "big")
                    ).digest(),
                    "big",
                ),
                Tensor._seed,
            ],
            device=device,
            dtype=dtypes.uint32,
            requires_grad=False,
        )
        Tensor._device_rng_counters[device] = Tensor(
            [num], device=device, dtype=dtypes.uint32, requires_grad=False
        )
    # increment rng counter for devices
    else:
        Tensor._device_rng_counters[device].assign(
            Tensor._device_rng_counters[device] + num
        ).contiguous()

    # threefry random bits
    bits_count = Tensor._device_rng_counters[device] - num
    counts0 = (
        Tensor.arange(
            ceildiv(num, 2), device=device, dtype=dtypes.uint32, requires_grad=False
        )
        + bits_count
    )
    counts1 = counts0 + ceildiv(num, 2)
    bits = Tensor._threefry_random_bits(Tensor._device_seeds[device], counts0, counts1)[
        :num
    ]

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
