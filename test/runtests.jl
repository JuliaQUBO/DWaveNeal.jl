using Test
using DWave
using DWaveNeal: DWaveNeal, MOI, QUBODrivers

@test DWaveNeal.MOI === DWave.Neal.MOI
@test DWaveNeal.QUBODrivers === DWave.Neal.QUBODrivers
@test DWaveNeal.Optimizer === DWave.Neal.Optimizer
@test DWaveNeal.neal === DWave.Neal.dwave_samplers

@test_logs (:warn, DWaveNeal._DEPRECATION_MESSAGE) DWaveNeal.__init__()

QUBODrivers.test(DWaveNeal.Optimizer)
