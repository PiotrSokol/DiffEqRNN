using DiffEqRNN
using OrdinaryDiffEq
using Random
using Test
using IterTools
using Flux, DiffEqFlux
using FiniteDifferences
using CUDA
CUDA.allowscalar(false)

@testset "Checking interpolation with CUDA: grid points" begin
    Random.seed!(0)
    bs = 16
    ts = 784
    A = randn(Float32, bs,ts)|> gpu
    τ = rand(1:(ts-1))|> gpu
    for itp ∈ [CubicSplineRegularGrid, LinearInterpolationRegularGrid, ConstantInterpolationRegularGrid]
        v = itp(A)
        @test isapprox(v(τ-1), A[:,τ])
    end
end # interpolation

@testset "Checking interpolation with CUDA: random points" begin
    Random.seed!(0)
    bs = 16
    ts = 784
    A = randn(Float32, ts)|> gpu
    τs = (ts-1)rand(100)
    v′ = CubicSpline(A |> cpu ,collect(0:ts-1))
    v = CubicSplineRegularGrid(repeat(A, 1,bs)|>permutedims);
    @testset "n-D spline" begin
      for τ ∈ τs
        @test isapprox(v(τ), gpu([v′(τ) for i in 1:bs ]))
      end
    end
    v′ = CubicSpline(A |> cpu ,collect(0:2:2ts-1))
    v = CubicSplineRegularGrid(repeat(A, 1,bs)|>permutedims,t₀=0,t₁=2ts-1,Δt=2);
    @testset "batched spline" begin
      for τ ∈ τs
        @test isapprox(v(τ), gpu([v′(τ) for i in 1:bs ]))
      end
    end
end

@testset "Checking batched CubicSpline with CUDA" begin
    Random.seed!(0)
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
    x = gpu.(x)
    times = gpu.(times)

    X = CubicSpline(x, times)
    X1d = CubicSpline.(cpu.(x),cpu.(times))
    tmax = minimum(maximum.(times))
    τs = (tmax-1)rand(100)
    for τ ∈ τs
      @test isapprox(X(τ), gpu([x(τ) for x ∈ X1d]),rtol=1e-3)
    end
end


@testset "Derivative tests with CUDA" begin
    Random.seed!(0)
    bs = 16
    ts = 784
    A = randn(Float32, bs,ts) |> gpu
    τs = 784rand(100)
    for itp ∈ [CubicSplineRegularGrid, LinearInterpolationRegularGrid]
        v = itp(A)
        for τ ∈ τs
          @test isapprox(central_fdm(5, 1)(t->v(t), τ), derivative(v,τ), rtol=1e-3)
        end
      end
end 

@testset "Batched derivative tests with CUDA" begin
    Random.seed!(0)
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

    x = gpu.(x)
    times = gpu.(times)
    tmax = minimum(maximum.(times))
    τs = (tmax-1)rand(100)
    X = CubicSpline(x, times)
    X1d = CubicSpline.(cpu.(x),cpu.(times))
    for τ ∈ τs
      @test isapprox(central_fdm(5, 1)(t->X(t), τ), derivative(X,τ), rtol=1e-3)
    end
end

@testset "Checking initial value problem for RNN ODE's  with CUDA" begin
    t₁ = 100
    for cell ∈ [∂RNNCell, ∂GRUCell, ∂LSTMCell] 
        ∂nn = cell(1,2) |> gpu
        tspan = Float32.([0, t₁])
        tsteps = collect(tspan[1] : tspan[2])
        node = RNNODE(∂nn, tspan, AutoTsit5(Rosenbrock23()), saveat=tsteps)
        # reltol=1e-8,abstol=1e-8
        sol = node(node.u₀)
        @test sol.retcode == :Success
    end
end

@testset "Checking inhomogeneous solution with CUDA: forward" begin
    Random.seed!(0)
    t₁ = 100
    bs = 7
    x = sqrt(1/2)randn(Float32, bs, t₁) |> gpu
    cells = [∂RNNCell, ∂GRUCell, ∂LSTMCell]
    interpolators = [CubicSplineRegularGrid, LinearInterpolationRegularGrid, ConstantInterpolationRegularGrid]

    for (cell, itp) ∈ Iterators.product(cells, interpolators)
        X = itp(x)
        ∂nn = cell(1,2) |> gpu
        tspan = Float32.([0, t₁])
        tsteps = collect(tspan[1] : tspan[2])
        node = RNNODE(∂nn, tspan, AutoTsit5(Rosenbrock23()), saveat=tsteps, preprocess=permutedims )
        # reltol=1e-8,abstol=1e-8
        sol = node(X)
        @test sol.retcode == :Success
    end
end

@testset "Checking inhomogeneous solution with CUDA: optimization" begin
    Random.seed!(0)
    t₁ = 10
    bs = 7
    x = sqrt(1/2)randn(Float32, bs, t₁) |> gpu
    cells = [∂RNNCell, ∂GRUCell, ∂LSTMCell]
    interpolators = [CubicSplineRegularGrid, LinearInterpolationRegularGrid, ConstantInterpolationRegularGrid]

    for (cell, itp) ∈ Iterators.product(cells, interpolators)
        X = itp(x)
        ∂nn = cell(1,2) |> gpu
        tspan = Float32.([0, t₁])
        tsteps = collect(tspan[1] : tspan[2])
        node = RNNODE(∂nn, tspan, AutoTsit5(Rosenbrock23()), reltol=1e-4,abstol=1e-4, saveat=tsteps, preprocess=permutedims )
        # reltol=1e-8,abstol=1e-8
        predict_neuralode(p) = gpu(node(X, p=p))
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
@testset "Neural CDE test with CUDA" begin
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
    X = CubicSpline(gpu.(x), gpu.(times))
    tmin = maximum(minimum.(times))
    tmax = minimum(maximum.(times))
    tspan = (tmin,tmax)
    ##
    inputsize+=1
    hiddensize = 16
    cde = Chain(
    Dense(hiddensize, hiddensize, celu),
    Dense(hiddensize, hiddensize*inputsize, tanh)) |> gpu
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
    result_neuralode = DiffEqFlux.sciml_train(loss_neuralcde, ncde.p, optim, maxiters = 100)
    @test result_neuralode.minimum < loss_before
end