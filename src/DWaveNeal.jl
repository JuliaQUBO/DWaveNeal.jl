module DWaveNeal

using PythonCall
import QUBODrivers:
    MOI,
    QUBODrivers,
    QUBOTools,
    Sample,
    SampleSet,
    @setup,
    sample,
    qubo

# -*- :: Python D-Wave Simulated Annealing :: -*- #
const neal = PythonCall.pynew() # initially NULL

function __init__()
    PythonCall.pycopy!(neal, pyimport("neal"))
end

const _DWAVE_NEAL_ATTR_LIST = [
    :num_reads,
    :num_sweeps,
    :num_sweeps_per_beta,
    :beta_range,
    :beta_schedule,
    :beta_schedule_type,
    :seed,
    :initial_states_generator,
    :interrupt_function,
]

@setup Optimizer begin
    name       = "D-Wave Neal Simulated Annealing Sampler"
    sense      = :min
    domain     = :bool
    version    = v"0.6.0"
    attributes = begin
        "num_reads"::Integer = 1_000
        "num_sweeps"::Integer = 1_000
        "num_sweeps_per_beta"::Integer = 1
        "beta_range"::Union{Tuple{Float64,Float64},Nothing} = nothing
        "beta_schedule"::Union{Vector,Nothing} = nothing
        "beta_schedule_type"::String = "geometric"
        "seed"::Union{Integer,Nothing} = nothing
        "initial_states_generator"::String = "random"
        "interrupt_function"::Union{Function,Nothing} = nothing
    end
end

@doc raw"""
    DWaveNeal.Optimizer{T}()

D-Wave's Simulated Annealing Sampler for QUBO and Ising models.
"""
Optimizer

function sample(sampler::Optimizer{T}) where {T}
    # Retrieve QUBO Model
    Q, α, β = qubo(sampler, Dict)

    # Retrieve Optimizer attributes
    n = MOI.get(sampler, MOI.NumberOfVariables())

    # Solver-specific attributes
    params = Dict{Symbol,Any}(
        param => MOI.get(sampler, MOI.RawOptimizerAttribute(string(param)))
        for param in _DWAVE_NEAL_ATTR_LIST
    )


    # Call D-Wave Neal API
    sampler = neal.SimulatedAnnealingSampler()
    results = @timed sampler.sample_qubo(Q; params...)
    var_map = pyconvert.(Int, results.value.variables)

    # Format Samples
    samples = Vector{Sample{T,Int}}()

    for (ϕ, λ, r) in results.value.record
        # dwave-neal will not consider variables that are not present
        # in the objective funcion, leading to holes with respect to
        # the indices in the record table.
        # Therefore, it is necessary to introduce an extra layer of
        # indirection to account for the missing variables.
        ψ = zeros(Int, n)

        for (i, v) in enumerate(ϕ)
            ψ[var_map[i]] = pyconvert(Int, v)
        end

        s = Sample{T,Int}(
            # state:
            ψ,
            # energy:
            α * (pyconvert(T, λ) + β),
            # reads:
            pyconvert(Int, r),
        )
        
        push!(samples, s)
    end

    # Write metadata
    metadata = Dict{String,Any}(
        "origin" => "D-Wave Neal",
        "time"   => Dict{String,Any}(
            "effective" => results.time
        ),
    )

    return SampleSet{T,Int}(samples, metadata)
end

end # module
