module DWaveNeal

import Anneal
using PythonCall
using MathOptInterface
const MOI = MathOptInterface

# -*- :: Python D-Wave Simulated Annealing :: -*- #
const neal = PythonCall.pynew() # initially NULL

function __init__()
    PythonCall.pycopy!(neal, pyimport("neal"))
end

Anneal.@anew Optimizer begin
    name = "D-Wave Neal Simulated Annealing Sampler"
    version = v"0.5.8"
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

const PARAMS = [
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
    # ~ Retrieve Ising Model ~ #
    _, Q, α, β = Anneal.qubo(sampler)

    # ~ Instantiate Sampler (Python) ~ #
    neal_sampler = neal.SimulatedAnnealingSampler()

    # ~ Retrieve Optimizer Attributes ~ #
    params = Dict{Symbol,Any}(
        param => MOI.get(
            sampler,
            MOI.RawOptimizerAttribute(string(param))
        )
        for param in PARAMS
    )

    # ~ Sample! ~ #
    results = @timed neal_sampler.sample_qubo(Q; params...)
    
    # ~ Basic Data Formatting ~ #
    records = results.value.record
    samples = Anneal.Sample{Int,T}[
        Anneal.Sample{Int,T}(
            pyconvert.(Int, ψ),        # state
            pyconvert(Int, n),         # reads
            α * (pyconvert(T, e) + β), # value
        )
        for (ψ, e, n) in records
    ]

    # ~ Write metadata ~ #
    metadata = Dict{String,Any}(
        "time" => Dict{String,Any}(
            "sample" => results.time
        )
    )

    # ~ Build SampleSet ~ #
    return Anneal.SampleSet{Int,T}(samples, metadata)
end

end # module
