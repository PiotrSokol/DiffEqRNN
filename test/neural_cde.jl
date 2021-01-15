using DiffEqRNN
using DataInterpolations
using DataInterpolations:derivative
using OrdinaryDiffEq
using Random
using Test
using IterTools
using Flux, DiffEqFlux
using DiffEqSensitivity
using Zygote
##
@testset "Neural CDE test" begin
    ##
    Random.seed!(2)
    t₁ = 10
    bs = 7
    inputsize = 3
    x = Float32(sqrt(1/2))randn(Float32, inputsize, bs, t₁)
    times = cumsum(randexp(Float32, 1, bs, t₁), dims=3)
    x2 = repeat(vcat([cos.(rand()*2π.+times[:,i,:]) for i in 1:size(times,2)]...),inner=(inputsize,1))
    x2 = reshape(x2, size(x))
    x = x2 .+ 0.1.*x
    x = Float32.(cat(times,x,dims=1))
    x = reshape(x, :,t₁)
    times = reshape(times, :, t₁)
    x = [x[i,:] for i ∈ 1:size(x,1)]
    times = repeat(times, inner=(inputsize+1,1))
    times = [times[i,:] for i ∈ 1:size(x,1)]
    tmax = minimum(maximum.(times))
    X = CubicSpline(x, times)
    tmin = maximum(minimum.(times))
    tmax = minimum(maximum.(times))
    tspan = (tmin,tmax)
    ##
    inputsize+=1
    hiddensize = 16
    cde = Chain(
    Dense(hiddensize, hiddensize, celu),
    Dense(hiddensize, hiddensize*inputsize, tanh))
    ncde = NeuralCDE(cde, tspan, inputsize, hiddensize, Tsit5(), reltol=1e-2,abstol=1e-2, preprocess=x->Float32.(reshape(x,1, inputsize, :)), sense = InterpolatingAdjoint(autojacvec=ZygoteVJP()) )

    sol = ncde(X)
    ##
    predict_neuralcde(p) = Array(ncde(X, p=p))
    function loss_neuralcde(p)
        pred = predict_neuralcde(p)
        loss = sum(abs2, pred[:,:,end] .- 0.0)
        return loss
    end
    loss_before = loss_neuralcde(ncde.p)
    optim = ADAM(0.05)
    result_neuralcde = DiffEqFlux.sciml_train(loss_neuralcde, ncde.p, optim, maxiters = 100)
    @test result_neuralcde.minimum < loss_before
end


# if Flux.use_cuda[]
# else
#   @warn "CUDA unavailable, not testing GPU support for NeuralCDE's"
# end
