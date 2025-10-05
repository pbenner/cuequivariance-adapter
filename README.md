# cuequivariance-adapter

This package provides Haiku/JAX wrappers that expose cuequivariance-backed
operations (linear layers, channel-wise tensor products, fully connected tensor
products) behind familiar e3nn-style interfaces.  Each adapter mirrors the
signature and layout conventions of the corresponding :mod:`e3nn.o3` module,
allowing existing e3nn pipelines to swap in cuequivariance implementations with
minimal code changes.

## Features

- **Linear** – Implements :math:`y = W x` with weights constrained by the
  Clebsch–Gordan rules.  Inputs/outputs are kept in e3nn's ``mul_ir`` layout,
  while cuequivariance performs the segmented-polynomial evaluation under the
  hood.
- **TensorProduct** – Channel-wise tensor product matching
  :class:`e3nn.o3.TensorProduct` (``'uvu'`` instructions).  Multiplicity handling
  and weight semantics follow e3nn conventions; cue handles the heavy lifting in
  ``ir_mul`` layout.
- **FullyConnectedTensorProduct** – Exhaustively mixes input multiplicities just
  like :class:`e3nn.o3.FullyConnectedTensorProduct`, delegating the computation
  to cue while preserving the e3nn UX.
- **SymmetricContraction** – Implements the species-dependent symmetric
  contraction used in MACE with the option to operate in the reduced CG basis or
  emulate the original formulation.

The adapters convert between the layout conventions automatically, collapse the
extra multiplicity axes introduced by cue descriptors, normalise results to
match e3nn, and manage shared or internal weights the way e3nn users expect.

## Installation

```bash
pip install -e .
```

Ensure the required dependencies listed in ``pyproject.toml`` (including
``cuequivariance`` and ``cuequivariance-jax``) are available in your environment.

## Running the tests

Tests compare the adapters against the PyTorch e3nn and cuequivariance-torch
implementations to guarantee numerical parity.

```bash
python -m pytest
```

## License

This project is distributed under the MIT License.  See ``LICENSE`` for details.
