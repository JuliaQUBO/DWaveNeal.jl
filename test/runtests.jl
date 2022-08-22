using Test
using Anneal
using DWaveNeal

function main()
    Anneal.test(DWaveNeal.Optimizer)
end

main() # Here we go!