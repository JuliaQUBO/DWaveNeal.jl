module DWaveNeal

using Anneal
using PythonCall

# -*- :: Python D-Wave Simulated Annealing :: -*- #
const neal = PythonCall.pynew() # initially NULL

function __init__()
    PythonCall.pycopy!(neal, pyimport("neal"))
end

Anneal.@anew Optimizer begin
    name       = "D-Wave Neal Simulated Annealing Sampler"
    sense      = :min
    domain     = :bool
    version    = v"0.5.9"
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

const PARAM_LIST = [
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

function Anneal.sample(sampler::Optimizer{T}) where {T}
    # ~*~ Retrieve Ising Model ~*~ #
    Q, α, β = Anneal.qubo(sampler, Dict, T)

    # ~*~ Retrieve Optimizer Attributes ~*~ #
    params = Dict{Symbol,Any}(
        param => MOI.get(
            sampler,
            MOI.RawOptimizerAttribute(string(param))
        )
        for param in PARAM_LIST
    )

    # ~*~ Call D-Wave Neal API ~*~ #
    results = @timed neal_sample(Q, α, β; params...)
    samples = results.value

    # ~*~ Timing Information ~*~ #
    time_data = Dict{String,Any}(
        "effective" => results.time
    )

    # ~*~ Write metadata ~*~ #
    metadata = Dict{String,Any}(
        "time"   => time_data,
        "origin" => "D-Wave Neal"
    )

    return Anneal.SampleSet{T}(samples, metadata)
end

function neal_sample(Q::Dict{Tuple{Int,Int},T}, α::T, β::T; params...) where {T}
    sampler = neal.SimulatedAnnealingSampler()
    records = sampler.sample_qubo(Q; params...).record
    samples = [
        Anneal.Sample{T,Int}(
            # state:
            pyconvert.(Int, ψ),
            # value: 
            α * (pyconvert(T, λ) + β),
            # reads:
            pyconvert(Int, r),        
        )
        for (ψ, λ, r) in records
    ]
    
    return samples
end

end # module
