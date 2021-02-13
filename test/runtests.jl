using DiffEqRNN
using Test
using Random

@assert isempty(ARGS) || "adj" ∈ ARGS

if isempty(ARGS)
  @testset "Initializers" begin
    include("initializers.jl")
  end
end

if isempty(ARGS)
  @testset "Interpolation tool" begin
    include("interp.jl")
  end
end

if isempty(ARGS)
  @testset "Continuous time RNN layers" begin
    include("fast_la.jl")
  end
end

@testset "Continuous time RNN solve & minimization" begin
  include("fast_rnn_ode.jl")
end

if isempty(ARGS)
   include("neural_cde.jl")
end

if Flux.use_cuda[]
  let to_run = to_run
    include("gpu.jl")
  end
else
  @warn "CUDA unavailable, not testing GPU support"
end