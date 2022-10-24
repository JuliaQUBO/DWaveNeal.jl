using Test
using DWaveNeal

function main()
    DWaveNeal.Anneal.test(DWaveNeal.Optimizer; examples=true)
end

main() # Here we go!