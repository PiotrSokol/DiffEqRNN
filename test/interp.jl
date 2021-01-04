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

@testset "Checking interpolation" begin
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

@testset "Flux.Data.Dataloader interface" begin
    ds = 16
    bs = 8
    ts = 784
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