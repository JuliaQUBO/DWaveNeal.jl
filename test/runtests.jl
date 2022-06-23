using Test
using Anneal
using DWaveNeal

function main()
    Anneal.@test_optimizer DWaveNeal.Optimizer
end

main() # Here we go!