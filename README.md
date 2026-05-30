# DWaveNeal.jl
> :warning: Deprecated compatibility package. New code should use [DWave.jl](https://github.com/JuliaQUBO/DWave.jl) and `DWave.Neal.Optimizer` directly.


[![DOI](https://zenodo.org/badge/506537248.svg)](https://zenodo.org/badge/latestdoi/506537248)
[![QUBODRIVERS](https://img.shields.io/badge/Powered%20by-QUBODrivers.jl-%20%234063d8)](https://github.com/JuliaQUBO/QUBODrivers.jl)


[D-Wave Neal](https://docs.ocean.dwavesys.com/projects/neal/en/latest/) compatibility layer for JuMP.

## Maintenance status
This package is maintained only as a compatibility shim for existing
`DWaveNeal.jl` users. New sampler features, bug fixes, and interface work should
happen in [DWave.jl](https://github.com/JuliaQUBO/DWave.jl).

## Installation
```julia
julia> import Pkg

julia> Pkg.add("DWave")

julia> using DWave
```

For existing environments that still depend on `DWaveNeal`, the compatibility package remains available:

```julia
julia> import Pkg

julia> Pkg.add("DWaveNeal")

julia> using DWaveNeal
```

`DWaveNeal.Optimizer` is an alias for `DWave.Neal.Optimizer` and will emit a deprecation warning on load.

## Getting started
```julia
using JuMP
using DWave

model = Model(DWave.Neal.Optimizer)

n = 3
Q = [ -1  2  2
       2 -1  2
       2  2 -1 ]

@variable(model, x[1:n], Bin)
@objective(model, Min, x' * Q * x)

optimize!(model)

for i = 1:result_count(model)
    xi = value.(model[:x]; result = i)
    yi = objective_value(model; result = i)

    println("[$i] f($(xi)) = $(yi)")
end
```

Legacy code that still does `Model(DWaveNeal.Optimizer)` should continue to work after upgrading to this package version.

**Note**: _The D-Wave Julia wrappers are not officially supported by D-Wave Systems. If you are a commercial customer interested in official support for Julia from D-Wave, let them know._
