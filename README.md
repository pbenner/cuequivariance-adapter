# cuequivariance-adapter

This package exposes cuequivariance-backed operations through both **Flax** and
**Haiku** modules with APIs that match the equivalent `e3nn.o3` components. By
swapping the original e3nn classes for these adapters you obtain the JAX based
cue implementations while keeping the familiar `mul_ir` layout and weight
semantics.

## What’s included?

- **Linear** – Layout-preserving linear map with optional shared or internal
  weights.
- **TensorProduct** – Channel-wise tensor product (``'uvu'`` instructions) that
  collapses cue’s expanded multiplicity axes back to the e3nn layout.
- **FullyConnectedTensorProduct** – Fully connected tensor product mirroring the
  behaviour of `e3nn.o3.FullyConnectedTensorProduct`.
- **SymmetricContraction** – Implementation of the MACE-style symmetric
  contraction supporting both the reduced and original CG bases.

Each adapter has a Flax module (``cuequivariance_adapter.flax``) and a Haiku
module (``cuequivariance_adapter.haiku``) with matching signatures. Under the
hood we convert between ``mul_ir`` and cue’s ``ir_mul`` layout, reshape and
normalise outputs, and delegate the segmented-polynomial evaluation to
`cuequivariance-jax`.

## Installation

We recommend creating a fresh virtual environment with JAX, Flax, Haiku and the
cue libraries. After cloning this repository run:

```bash
pip install -e .
```

This installs the adapters in editable mode using the dependency versions
specified in `pyproject.toml`. The test suite also requires PyTorch,
`cuequivariance-torch`, and `e3nn` for cross-checks.

## Quick usage examples

### Flax

```python
from e3nn import o3
from cuequivariance_adapter.flax import Linear

linear = Linear(o3.Irreps('2x0e + 1x1o'), o3.Irreps('3x0e'))
params = linear.init(jax.random.PRNGKey(0), jnp.zeros((4, linear.irreps_in.dim)))
out = linear.apply(params, jnp.ones((4, linear.irreps_in.dim)))
```

### Haiku

```python
import haiku as hk
from e3nn import o3
from cuequivariance_adapter.haiku import TensorProduct

def forward(x1, x2):
    tp = TensorProduct(o3.Irreps('1x0e + 1x1o'), o3.Irreps('1x0e'), o3.Irreps('2x0e'))
    return tp(x1, x2)

apply_fn = hk.transform(forward).apply
```

## Running the tests

The tests compare each adapter against the e3nn reference implementation and
the equivalent cuequivariance/PyTorch adapter to guarantee numerical agreement.

```bash
pytest
```

The tests emit a few deprecation warnings from Haiku/JAX in some configurations;
these are harmless and can be ignored.

## License

This project is distributed under the MIT License. See `LICENSE` for details.
