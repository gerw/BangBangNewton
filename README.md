[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21239638.svg)](https://doi.org/10.5281/zenodo.21239638)

# Continuous differentiability of the signum function and Newton’s method for bang-bang control

This repository contains the implementation of the two variants of Newton's
method to solve bang-bang optimal control problems as described in the paper
"Continuous differentiability of the signum function and Newton’s method for
bang-bang control" by Daniel Wachsmuth and Gerd Wachsmuth.

To run the examples from the paper, start `julia` and use
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate() # Only needed once.

include("paper.jl")
solve_all_examples(3) # Solve small instances to precompile everything
solve_all_examples()  # Solve the large instances
```
The code was tested with `julia` version 1.11.4.

# Citation
If you find this helpful, please cite the paper and the source code as
```bibtex
@online{WachsmuthWachsmuth2025,
  author        = {Daniel Wachsmuth and Gerd Wachsmuth},
  title         = {Continuous differentiability of the signum function and Newton's method for bang-bang control},
  year          = {2025},
  eprint        = {2509.24829},
  eprinttype    = {arXiv}
}

@misc{WachsmuthWachsmuth2026,
    author = {Daniel Wachsmuth and Gerd Wachsmuth},
    title = {Continuous differentiability of the signum function and Newton's method for bang-bang control},
    year = {2026},
    doi = {10.5281/zenodo.21239638}
}
```

