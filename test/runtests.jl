using DiffEqRNN
using Test
using Random

@testset "Initializers" begin
  include("initializers.jl")
end

@testset "Interpolation tool" begin
  include("interp.jl")
end

@testset "Continuous time RNN layers" begin
  include("layers.jl")
end

@testset "Continuous time RNN solve & minimization" begin
  include("rnn_ode.jl")
end

# @testset "CUDA RNN ODE and NeuralCDE test" begin
#   include("gpu.jl")
# end