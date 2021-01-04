using DiffEqRNN
using OrdinaryDiffEq
using Random
using Test
using IterTools
using Flux, DiffEqFlux

@testset "Checking initial value problem for RNN ODE's" begin
    t₁ = 100
    for cell ∈ [∂RNNCell, ∂GRUCell, ∂LSTMCell] 
        ∂nn = cell(1,2)
        tspan = Float32.([0, t₁])
        tsteps = collect(tspan[1] : tspan[2])
        node = RNNODE(∂nn, tspan, AutoTsit5(Rosenbrock23()), saveat=tsteps, preprocess=permutedims )
        # reltol=1e-8,abstol=1e-8
        sol = node(node.u₀)
        @test sol.retcode == :Success
    end
end

@testset "Checking inhomogeneous solution" begin
    Random.seed!(0)
    t₁ = 100
    bs = 7
    x = sqrt(1/2)randn(Float32, bs, t₁)
    cells = [∂RNNCell, ∂GRUCell, ∂LSTMCell]
    interpolators = [CubicSplineRegularGrid, LinearInterpolationRegularGrid, ConstantInterpolationRegularGrid]

    for (cell, itp) ∈ product(cells, interpolators)
        X = itp(x)
        ∂nn = cell(1,2)
        tspan = Float32.([0, t₁])
        tsteps = collect(tspan[1] : tspan[2])
        node = RNNODE(∂nn, tspan, AutoTsit5(Rosenbrock23()), saveat=tsteps, preprocess=permutedims )
        # reltol=1e-8,abstol=1e-8
        sol = node(X)
        @test sol.retcode == :Success
    end
end

@testset "Checking inhomogeneous solution" begin
    Random.seed!(0)
    t₁ = 10
    bs = 7
    x = sqrt(1/2)randn(Float32, bs, t₁)
    cells = [∂RNNCell, ∂GRUCell, ∂LSTMCell]
    interpolators = [CubicSplineRegularGrid, LinearInterpolationRegularGrid, ConstantInterpolationRegularGrid]

    for (cell, itp) ∈ IterTools.product(cells, interpolators)
        X = itp(x)
        ∂nn = cell(1,2)
        tspan = Float32.([0, t₁])
        tsteps = collect(tspan[1] : tspan[2])
        node = RNNODE(∂nn, tspan, AutoTsit5(Rosenbrock23()), reltol=1e-4,abstol=1e-4, saveat=tsteps, preprocess=permutedims )
        # reltol=1e-8,abstol=1e-8
        predict_neuralode(p) = Array(node(X, p=p))
        function loss_neuralode(p)
            pred = predict_neuralode(p)
            loss = sum(abs2, pred .- 0.0)
            return loss
        end
        optim = ADAM(0.05)
        result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, node.p, optim, maxiters = 100)
        @test result_neuralode.ls_success
    end
end