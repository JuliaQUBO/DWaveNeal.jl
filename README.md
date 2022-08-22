# DWaveNeal.jl
DWave Neal Simulated Annealing Interface for JuMP

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

@objective(model, Min, x' * Q * x)

optimize!(model)

for i = 1:result_count(model)
    x = value.(model; result = i)
    y = objective_value(model; result = i)

    println("f($(x)) = $(y)")
end
```