using DiffEqRNN
using OrdinaryDiffEq
using Random
using Test
using IterTools
using Flux, DiffEqFlux

# @testset "Checking inhomogeneous solution" begin
    Random.seed!(0)
    t₁ = 10
    bs = 7
    inputsize = 3
    x = sqrt(1/2)randn(Float32, inputsize, bs, t₁)
    times = cumsum(randexp(Float32, 1, bs, t₁), dims=3)
    x = cat(times,x,dims=1)
    x = reshape(x, :,t₁)
    times = reshape(times, :, t₁)
    x = [x[i,:] for i ∈ 1:size(x,1)]
    times = repeat(times, inner=(inputsize+1,1))
    times = [times[i,:] for i ∈ 1:size(x,1)]
    X = CubicSpline(x, times)
    # cells = [∂RNNCell, ∂GRUCell, ∂LSTMCell]

    # interpolators = [CubicSplineRegularGrid, LinearInterpolationRegularGrid, ConstantInterpolationRegularGrid]

    # for (cell, itp) ∈ IterTools.product(cells, interpolators)
    #     X = itp(x)
    #     ∂nn = cell(1,2)
    #     tspan = Float32.([0, t₁])
    #     tsteps = collect(tspan[1] : tspan[2])
    #     node = RNNODE(∂nn, tspan, AutoTsit5(Rosenbrock23()), reltol=1e-4,abstol=1e-4, saveat=tsteps, preprocess=permutedims )
    #     # reltol=1e-8,abstol=1e-8
    #     predict_neuralode(p) = Array(node(X, p=p))
    #     function loss_neuralode(p)
    #         pred = predict_neuralode(p)
    #         loss = sum(abs2, pred .- 0.0)
    #         return loss
    #     end
    #     optim = ADAM(0.05)
    #     result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, node.p, optim, maxiters = 100)
    #     @test result_neuralode.ls_success
    # end
# end