module DWaveNeal

import DWave

@doc raw"""
    DWaveNeal.Optimizer{T}()

D-Wave Neal compatibility wrapper over `DWave.Neal.Optimizer`.
"""
const MOI = DWave.Neal.MOI
const QUBODrivers = DWave.Neal.QUBODrivers
const QUBOTools = DWave.Neal.QUBOTools
const PythonCall = DWave.Neal.PythonCall
const neal = DWave.Neal.dwave_samplers
const Optimizer = DWave.Neal.Optimizer

const _DEPRECATION_MESSAGE = """
DWaveNeal.jl is deprecated. Use DWave.jl instead.

For new code:
    using DWave
    model = Model(DWave.Neal.Optimizer)

DWaveNeal.Optimizer currently aliases DWave.Neal.Optimizer for compatibility.
"""

function __init__()
    @warn _DEPRECATION_MESSAGE maxlog = 1

    return nothing
end

end # module
