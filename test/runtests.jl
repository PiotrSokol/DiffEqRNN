using DiffEqRNN
using Test
using Random
using Flux

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

include("neural_cde.jl")

if Flux.use_cuda[]
  include("gpu.jl")
else
  @warn "CUDA unavailable, not testing GPU support"
end