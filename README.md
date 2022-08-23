# DWaveNeal.jl
D-Wave Neal Simulated Annealing Interface for JuMP

```julia
julia> import Pkg; Pkg.add("DWaveNeal")

julia> using DWaveNeal
```

```julia
using JuMP
using DWaveNeal
```

```julia
model = Model(DWaveNeal.Optimizer)

n = 3
Q = 2 * rand(n, n) .- 1.0 # Q[i, j] in [-1, 1]

@variable(model, x[1:n], Bin)
@objective(model, Min, x' * Q * x)

optimize!(model)

for i = 1:result_count(model)
    x = value.(model; result = i)
    y = objective_value(model; result = i)

    println("f($(x)) = $(y)")
end
```

**Note**: _The D-Wave Neal wrapper for Julia is not officially supported by D-Wave Systems. If you are a commercial customer interested in official support for Julia from DWave, let them know!_