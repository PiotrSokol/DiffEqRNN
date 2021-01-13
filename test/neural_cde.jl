using DiffEqRNN
using OrdinaryDiffEq
using Random
using Test
using IterTools
using Flux, DiffEqFlux
using DiffEqSensitivity
using Zygote

@testset "Neural CDE test" begin
    Random.seed!(2)
    t₁ = 10
    bs = 7
    inputsize = 3
    x = Float32(sqrt(1/2))randn(Float32, inputsize, bs, t₁)
    times = cumsum(randexp(Float32, 1, bs, t₁), dims=3)
    x = cat(times,x,dims=1)
    x = reshape(x, :,t₁)
    times = reshape(times, :, t₁)
    x = [x[i,:] for i ∈ 1:size(x,1)]
    times = repeat(times, inner=(inputsize+1,1))
    times = [times[i,:] for i ∈ 1:size(x,1)]
    tmax = minimum(maximum.(times))
    X = CubicSpline(x, times)
    tmin = maximum(minimum.(times))
    tspan = (tmin,tmax)
    tsteps = collect(tspan[1] : tspan[2])

    inputsize+=1
    hiddensize = 16
    cde = Chain(
    Dense(hiddensize, hiddensize, relu),
    Dense(hiddensize, hiddensize*inputsize, x->tanh(x).-x),)
    ncde = NeuralCDE(cde, tspan, inputsize, hiddensize, Tsit5(), reltol=1e-4,abstol=1e-4, preprocess=x->reshape(x,1, inputsize, :), sense = InterpolatingAdjoint(autojacvec=ZygoteVJP()) )

    sol = ncde(X)
    predict_neuralcde(p) = Array(ncde(X, p=p))
    function loss_neuralcde(p)
        pred = predict_neuralcde(p)
        loss = sum(abs2, pred[:,:,end] .- 0.0)
        return loss
    end
    optim = ADAM(0.05)
    result_neuralode = DiffEqFlux.sciml_train(loss_neuralcde, ncde.p, optim, maxiters = 1)
    @test result_neuralode.ls_success
end


if Flux.use_cuda[]
    @testset "Neural CDE test" begin
    Random.seed!(0)
    t₁ = 10
    bs = 7
    inputsize = 3
    x = Float32(sqrt(1/2))randn(Float32, inputsize, bs, t₁) |>gpu
    times = cumsum(randexp(Float32, 1, bs, t₁), dims=3) |>gpu
    x = cat(times,x,dims=1)
    x = reshape(x, :,t₁)
    times = reshape(times, :, t₁)
    x = [x[i,:] for i ∈ 1:size(x,1)]
    times = repeat(times, inner=(inputsize+1,1))
    times = [times[i,:] for i ∈ 1:size(x,1)]
    tmax = maximum(maximum.(times)) |> cpu
    X = CubicSpline(x, times)
    
    tspan = (0.f0,tmax)
    tsteps = collect(tspan[1] : tspan[2])


    hiddensize = 16
    cde = Chain(
    Dense(hiddensize, hiddensize, relu),
    Dense(hiddensize, hiddensize*inputsize, tanh),)|>gpu
    ncde = NeuralCDE(cde, tspan, AutoTsit5(Rosenbrock23()), reltol=1e-4,abstol=1e-4, saveat=tsteps, preprocess=x->reshape(x,1, inputsize, :) )

    predict_neuralode(p) = gpu(ncde(X, p=p))
    function loss_neuralode(p)
        pred = predict_neuralode(p)
        loss = sum(abs2, pred .- 0.0)
        return loss
    end
    optim = ADAM(0.05)
    result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, node.p, optim, maxiters = 100)
    @test result_neuralode.ls_success
    end
else
  @warn "CUDA unavailable, not testing GPU support"
end