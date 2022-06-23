module DWaveNeal

using Anneal
using PythonCall
using MathOptInterface
const MOI = MathOptInterface
const VI = MOI.VariableIndex
const ROW = MOI.RawOptimizerAttribute

# -*- :: Python D-Wave Simulated Annealing :: -*- #
const neal = PythonCall.pynew() # initially NULL

function __init__()
    PythonCall.pycopy!(neal, pyimport("neal"))
end

mutable struct Optimizer{T} <: Anneal.AbstractSampler{T}
    x::Dict{VI, Int}
    y::Dict{Int, VI}
    s::T
    Q::Dict{Tuple{Int, Int}, T}
    c::T

    attrs::Dict{String, Any}

    function Optimizer{T}() where T
        new{T}(
            Dict{VI, Int}(),
            Dict{Int, VI}(),
            one(T),
            Dict{Tuple{Int, Int}, T}(),
            zero(T),
            Dict{String, Any}(),
        )
    end
end

# -*- :: Attributes :: -*- #
# --> Raw attribute fallback
function raw_attr_str(::MOI.AbstractOptimizerAttribute)::String end
MOI.get(annealer::Optimizer, attr::MOI.AbstractOptimizerAttribute) = MOI.get(annealer, ROW(raw_attr_str(attr)))
MOI.get(annealer::Optimizer, attr::ROW) = MOI.get(annealer, attr, Val(attr.value))
MOI.set(annealer::Optimizer, attr::ROW, value::Any) = MOI.set(annealer, attr, Val(attr.value), value)

struct NumberOfReads <: MOI.AbstractOptimizerAttribute end
raw_attr(::NumberOfReads) = "num_reads"
MOI.get(annealer::Optimizer, ::ROW, ::Val{"num_reads"}) = (annealer.attrs["num_reads"]::Integer)
MOI.set(annealer::Optimizer, ::ROW, ::Val{"num_reads"}, value::Integer) = (annealer.attrs["num_reads"] = value)

struct NumberOfSweeps <: MOI.AbstractOptimizerAttribute end
raw_attr(::NumberOfReads) = "num_sweeps"
MOI.get(annealer::Optimizer, ::ROW, ::Val{"num_sweeps"}) = (annealer.attrs["num_sweeps"]::Integer)
MOI.set(annealer::Optimizer, ::ROW, ::Val{"num_sweeps"}, value::Integer) = (annealer.attrs["num_sweeps"] = value)

# -*- :: Anneal.jl Interface :: -*- #
MOI.get(annealer::Optimizer, ::Anneal.x) = annealer.x
MOI.get(annealer::Optimizer, ::Anneal.y) = annealer.y
MOI.get(annealer::Optimizer, ::Anneal.Q) = annealer.Q
MOI.get(annealer::Optimizer, ::Anneal.c) = annealer.c

function Anneal.sample(annealer::Optimizer{T}) where T
    s, Q, c = Anneal.qubo_normal_form(annealer)
    sampler = neal.SimulatedAnnealingSampler()
    records = sampler.sample_qubo(
        s * Q,
        num_reads = MOI.get(annealer, ROW("num_reads")),
        num_sweeps = MOI.get(annealer, ROW("num_sweeps")),
    ).record
    samples = [(
        pyconvert.(Int, ψ),
        pyconvert(Int, n),
        pyconvert(T, e) + c,
    ) for (ψ, e, n) in records]

    return samples
end

end # module
