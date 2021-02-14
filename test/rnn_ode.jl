using OrdinaryDiffEq
using Flux, DiffEqSensitivity, DiffEqFlux, Zygote

if isempty(ARGS)
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
    ##
    @testset "Checking inhomogeneous solution" begin
        Random.seed!(0)
        t₁ = 100
        bs = 7
        x = sqrt(1/2)randn(Float32, bs, t₁)
        cells = [∂RNNCell, ∂GRUCell, ∂LSTMCell]
        interpolators = [CubicSplineRegularGrid, LinearInterpolationRegularGrid, ConstantInterpolationRegularGrid]

        for (cell, itp) ∈ Iterators.product(cells, interpolators)
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
end
if "adj" ∈ ARGS
        @testset "Checking inhomogeneous solution: adjoint" begin
        Random.seed!(0)
        t₁ = 10
        bs = 7
        x = sqrt(1/2)randn(Float32, bs, t₁)
        cells = [∂RNNCell, ∂GRUCell, ∂LSTMCell]
        interpolators = [CubicSplineRegularGrid, LinearInterpolationRegularGrid, ConstantInterpolationRegularGrid]

        for (cell, itp) ∈ Iterators.product(cells, interpolators)
            X = itp(x)
            ∂nn = cell(1,2)
            tspan = Float32.([0, t₁])
            tsteps = collect(tspan[1] : tspan[2])
            node = RNNODE(∂nn, tspan, Tsit5(), reltol=1e-4,abstol=1e-4, saveat=tsteps, preprocess=permutedims )

            predict_neuralode(p) = Array(node(X,p))
            loss_neuralode(p) = sum(abs2, predict_neuralode(p))
            ∇ = Zygote.gradient( p-> loss_neuralode(p), node.p)
            @test typeof(∇) <: Tuple{<:VecOrMat{<:AbstractFloat}}
        end
    end
end

if isempty(ARGS)
    @testset "Checking inhomogeneous solution: optimization" begin
        Random.seed!(0)
        t₁ = 10
        bs = 7
        x = sqrt(1/2)randn(Float32, bs, t₁)
        cells = [∂RNNCell, ∂GRUCell, ∂LSTMCell]
        interpolators = [CubicSplineRegularGrid, LinearInterpolationRegularGrid, ConstantInterpolationRegularGrid]

        for (cell, itp) ∈ Iterators.product(cells, interpolators)
            X = itp(x)
            ∂nn = cell(1,2)
            tspan = Float32.([0, t₁])
            tsteps = collect(tspan[1] : tspan[2])
            node = RNNODE(∂nn, tspan, Tsit5(), reltol=1e-4,abstol=1e-4, saveat=tsteps, preprocess=permutedims )

            predict_neuralode(p) = Array(node(X,p))
            loss_neuralode(p) = sum(abs2, predict_neuralode(p))
            loss_before = loss_neuralode(node.p)
            optim = ADAM(0.05)
            result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, node.p, optim, maxiters = 100)
            @test result_neuralode.minimum < loss_before
        end
    end
end
