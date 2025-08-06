import dataclasses
import functools
import typing

from tinygrad import nn
from tinygrad import Tensor
from tinygrad import TinyJit

from .multi_nn import MultiModel


@dataclasses.dataclass
class Spec:
    model: MultiModel
    upstream_sampling: int = 0
    evolve: bool = True


def produce(
    model: MultiModel,
    x: Tensor,
    paths: Tensor | None = None,
    upstream_sampling: int = 0,
) -> tuple[Tensor, Tensor]:
    if paths is None:
        # this is the first spec for taking in the raw input, let's feed data to all of them
        output_data = Tensor.stack(
            *(model(Tensor(i), x) for i in range(model.vendor_count)), dim=0
        )
        paths = Tensor.arange(model.vendor_count).unsqueeze(1)
        return output_data, paths
    if x.size(0) != paths.size(0):
        raise ValueError(
            "Provided input data's first dimension doesn't match with the paths' first dimension"
        )

    if upstream_sampling == 0:
        upstream_sampling = x.shape[0]

    input_count = paths.size(0)
    input_indexes = Tensor.stack(
        *(
            Tensor.randperm(input_count)[:upstream_sampling]
            for _ in range(model.vendor_count)
        ),
        dim=0,
    )
    input_data = x[input_indexes]
    # merge different batches for the same vendor into one. not sure if this is needed, but at least it saves us
    # from calling the model multiple times and making the graph more complex
    merged_batches = input_data.reshape(input_data.shape[0], -1, *input_data.shape[3:])

    output_data = Tensor.stack(
        *(
            model(i, merged)
            for i, merged in zip(range(model.vendor_count), merged_batches)
        ),
        dim=0,
    )
    # breaking down merged batches back to individual batches
    output_data = output_data.reshape(-1, input_data.shape[2], *output_data.shape[2:])

    prev_paths = paths[input_indexes].flatten(0, 1)
    new_paths = (
        Tensor.arange(model.vendor_count)
        .unsqueeze(1)
        .repeat(1, upstream_sampling)
        .flatten()
        .unsqueeze(1)
    )
    merged_paths = prev_paths.cat(new_paths, dim=1)

    return output_data, merged_paths


def forward(
    specs: list[Spec], x: Tensor, initial_paths: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    def step(acc: tuple[Tensor, Tensor | None], spec: Spec) -> tuple[Tensor, Tensor]:
        data, paths = acc
        return produce(
            model=spec.model,
            x=data,
            paths=paths,
            upstream_sampling=spec.upstream_sampling,
        )

    return functools.reduce(step, specs, (x, initial_paths))


def uniform_between(
    lhs: Tensor,
    rhs: Tensor,
    jitter_scale: Tensor | None = None,
    jitter_offset: Tensor | None = None,
):
    if lhs.shape != rhs.shape:
        raise ValueError("Shape of two tensors should be the same")
    scale = Tensor.uniform(lhs.shape)
    delta = rhs - lhs
    base = lhs
    if jitter_scale is not None:
        base -= delta * jitter_scale
        delta *= 1 + jitter_scale * 2
    if jitter_offset is not None:
        base -= jitter_offset
        delta += jitter_offset * 2
    # Interpolate between two tensor
    return base + delta * scale


@TinyJit
def make_offsprings(
    profit_matrix: Tensor,
    marketplace: list[Spec],
    offspring_count: int,
    keep_count: int,
    jitter_scale: Tensor | None = None,
    jitter_offset: Tensor | None = None,
):
    for vendor_profits, spec in zip(profit_matrix, marketplace):
        if not spec.evolve:
            continue

        best_profits, mutate_indexes = vendor_profits.topk(
            offspring_count, largest=True
        )
        print(best_profits.tolist())
        new_params = []
        for src_idx in mutate_indexes:
            src = spec.vendors[src_idx.item()]
            src_params = nn.state.get_state_dict(src)
            new_params.append(
                {
                    key: (
                        src_params[key]
                        + Tensor.uniform(
                            *src_params[key].shape,
                            low=-jitter_offset,
                            high=jitter_offset,
                        )
                    ).realize()
                    for key in src_params
                }
            )

        phase_out_count = len(spec.vendors) - keep_count

        _, phase_out_indexes = vendor_profits.topk(phase_out_count, largest=False)
        for params, index in zip(new_params, phase_out_indexes):
            nn.state.load_state_dict(spec.vendors[index.item()], params, verbose=False)

        for idx in phase_out_indexes[offspring_count:]:
            spec.vendors[idx.item()] = spec.model_factory()


# @TinyJit
def mutate(marketplace: list[Spec], leading_path: Tensor, jitter: Tensor):
    for spec, leading_index in zip(marketplace, leading_path):
        if not spec.evolve:
            continue
        leading_index = leading_index.item()
        leading_params = nn.state.get_state_dict(spec.vendors[leading_index])
        for i, vendor in enumerate(spec.vendors):
            if i == leading_index:
                continue
            nn.state.load_state_dict(
                spec.vendors[i],
                {
                    key: (
                        leading_params[key]
                        + Tensor.uniform(
                            *leading_params[key].shape, low=-jitter, high=jitter
                        )
                    ).realize()
                    for key in leading_params
                },
                verbose=False,
            )
