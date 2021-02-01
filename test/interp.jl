import DiffEqRNN:CubicSplineRegularGrid, LinearInterpolationRegularGrid, ConstantInterpolationRegularGrid, derivative
using Test
using Random
using FiniteDifferences
import Flux.Data:DataLoader

@testset "Checking constructors & type stability" begin
    for shps ∈ [(200,),(200,10)]
    Random.seed!(0)
    A = randn(Float32,shps)
      for itp ∈ [CubicSplineRegularGrid, LinearInterpolationRegularGrid, ConstantInterpolationRegularGrid]
        v = itp(A)
        @test eltype(v) == Float32
      end
    end
  end # constructors


@testset "Checking interpolation: grid points" begin
    Random.seed!(0)
    bs = 16
    ts = 784
    A = randn(Float32, bs,ts)
    τ = rand(1:(ts-1))
    for itp ∈ [CubicSplineRegularGrid, LinearInterpolationRegularGrid, ConstantInterpolationRegularGrid]
        v = itp(A)
        @test isapprox(v(τ-1), A[:,τ])
    end
end # interpolation

@testset "Checking interpolation: random points" begin
    Random.seed!(0)
    bs = 16
    ts = 784
    A = randn(Float32, ts)
    τs = (ts-1)rand(100)
    v = CubicSplineRegularGrid(A)
    v′ = CubicSpline(A,collect(0:ts-1))
    @testset "1-D spline" begin
      for τ ∈ τs
        @test isapprox(v(τ), v′(τ))
      end
    end
    v = CubicSplineRegularGrid(repeat(A, 1,bs)|>permutedims)
    @testset "n-D spline" begin
      for τ ∈ τs
        @test isapprox(v(τ)[1], v′(τ))
      end
    end
    
    v = CubicSplineRegularGrid(A,t₀=0,t₁=2ts-1,Δt=2)
    v′ = CubicSpline(A,collect(0:2:2ts-1))
    for τ ∈ τs
      @test isapprox(v(τ), v′(τ))
    end
    v = CubicSplineRegularGrid(repeat(A, 1,bs)|>permutedims,t₀=0,t₁=2ts-1,Δt=2)
    for τ ∈ τs
      @test isapprox(v(τ)[1], v′(τ))
    end
end

@testset "Checking batched CubicSpline" begin
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
    X = CubicSpline(x, times)
    X1d = CubicSpline.(x,times)
    tmax = minimum(maximum.(times))
    τs = (tmax-1)rand(100)
    for τ ∈ τs
      @test isapprox(X(τ), [x(τ) for x ∈ X1d],rtol=1e-3)
    end
end


@testset "Derivative tests" begin
    Random.seed!(0)
    bs = 16
    ts = 784
    A = randn(Float32, bs,ts)
    τs = 784rand(100)
    for itp ∈ [CubicSplineRegularGrid, LinearInterpolationRegularGrid]
        v = itp(A)
        for τ ∈ τs
          @test isapprox(central_fdm(5, 1)(t->v(t), τ), derivative(v,τ), rtol=1e-3)
        end
      end
end 

@testset "Batched derivative tests" begin
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

    tmax = minimum(maximum.(times))
    τs = (tmax-1)rand(100)
    X = CubicSpline(x, times)
    X1d = CubicSpline.(x, times)
    for τ ∈ τs
      @test isapprox(central_fdm(5, 1)(t->X(t), τ), derivative(X,τ), rtol=1e-3)
    end
end 

@testset "Flux.Data.Dataloader interface" begin
    ds = 432
    bs = 216
    ts = 196
    x = randn(Float32, ds,ts)
    Y = randn(Float32, ds)
    for itp ∈ [CubicSplineRegularGrid, LinearInterpolationRegularGrid, ConstantInterpolationRegularGrid]
        X = itp(x)
        dl = DataLoader((X, Y); batchsize = bs, shuffle = false)
        x1, y1 =first(dl)
        @test x1.u == X.u[1:bs,:]
        @test x[1:bs,:] == x1.u
      end
end # interpolation

