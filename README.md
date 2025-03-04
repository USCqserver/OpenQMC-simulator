# OpenQMC-simulator
QMC implementation for open quantum dynamics

## Basic Usage
Basic working examples corresponding to the circuits simulated in [![Paper](https://img.shields.io/badge/paper-arXiv%3A2502.18929-B31B1B.svg)](https://arxiv.org/abs/2502.18929) are provided in the [examples](examples/) folder. Circuits and noise models must be constructed exclusively using [HOQST](https://github.com/USCqserver/OpenQuantumTools.jl), as this is the only supported method for defining simulation settings. 
### Download the Sysimage Asset
Download the asset (e.g. openqmc-simulator-macos.zip) from the GitHub release and extract it. The extracted file is the compiled simulator.
### Launch Julia Using the Sysimage
With your standard Julia installation and instantiation, start a new session that uses the sysimage. For example, on macOS:
```
export JULIA_NUM_THREADS=8
julia --project=@. --sysimage compiled/lib/julia/sys.dylib
```
Once the REPL starts with the custom sysimage, you can run the GHZ state preparation simulation script (or other examples) from the REPL.
```julia-REPL
julia> include("examples/GHZ_state_preparation.jl")
  4.021899 seconds (9.36 M allocations: 483.778 MiB, 1.97% gc time, 77.10% compilation time)
QMCState with 413 occupied states
```
The output is a sparse representation of the density matrix, $\rho$, where only statistically significant states (elements in the density matrix) are stored. You can access a specific state ID using:
```julia-REPL
julia> ρ(ID::Int)
```
For example, `ID = 1` state corresponds to the (1,1) element of the density matrix. To view the full list of states:
```julia-REPL
julia> ρ.pop_list
```
For a full-state tomography, use:
```julia-REPL
julia> reconstruct_state(ρ)
32×32 SparseMatrixCSC{ComplexF64, Int64} with 412 stored entries:
⎡⣿⣿⢕⢟⠑⢄⢕⢟⠟⠅⠁⠀⠟⠅⣵⣿⎤
⎢⣵⢕⣿⣿⠑⠄⠑⠄⠁⠀⠁⠀⣵⢕⢕⣵⎥
⎢⠑⢄⠑⠄⣿⣿⠕⠟⠑⠄⠑⠄⠑⠄⠑⠄⎥
⎢⣵⢕⠑⠄⣵⠅⢟⣵⣵⢕⠑⠄⠑⠄⢕⣵⎥
⎢⠑⠄⠀⠀⠑⠄⠅⠁⣿⣿⠑⢟⠑⢄⠅⠟⎥
⎢⠀⠀⠁⠀⠁⠀⠁⠀⣵⢅⢟⣥⠀⠀⠁⠀⎥
⎢⢟⠵⠕⠟⠑⠄⠑⠄⠑⢄⠁⠀⢟⣵⠕⢟⎥
⎣⣵⣿⢟⣵⠑⠄⢕⣵⣥⠅⠁⠀⡵⢅⣿⣵⎦
```
Note that full-state tomography can be computationally expensive for large systems. Therefore, it's often more efficient to perform local measurements that access specific elements in the density matrix.