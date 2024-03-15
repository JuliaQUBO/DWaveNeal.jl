module DWaveNeal

using Anneal # Exports MOI
using PythonCall

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

Anneal.@anew Optimizer begin
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
""" Optimizer

function Anneal.sample(sampler::Optimizer{T}) where {T}
    # ~*~ Retrieve Ising Model ~*~ #
    Q, α, β = Anneal.qubo(sampler, Dict)

    # ~*~ Retrieve Optimizer Attributes ~*~ #
    params = Dict{Symbol,Any}(
        param => MOI.get(sampler, MOI.RawOptimizerAttribute(string(param)))
        for param in _DWAVE_NEAL_ATTR_LIST
    )

    # ~*~ Call D-Wave Neal API ~*~ #
    sampler = neal.SimulatedAnnealingSampler()
    results = @timed sampler.sample_qubo(Q; params...)
    records = results.value.record

    # ~*~ Format Samples ~*~ #
    samples = [
        Anneal.Sample{T,Int}(
            # state:
            pyconvert.(Int, ψ),
            # value: 
            α * (pyconvert(T, λ) + β),
            # reads:
            pyconvert(Int, r),
        ) for (ψ, λ, r) in records
    ]

    # ~*~ Write metadata ~*~ #
    metadata = Dict{String,Any}(
        "origin" => "D-Wave Neal",
        "time"   => Dict{String,Any}(
            "effective" => results.time
        ),
    )

    return Anneal.SampleSet{T}(samples, metadata)
end

end # module
