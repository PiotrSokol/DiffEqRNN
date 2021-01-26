module DiffEqRNN

using DataInterpolations
import DataInterpolations: AbstractInterpolation, munge_data, derivative
import LinearAlgebra:Tridiagonal, qr, Diagonal
import Zygote:ignore, Buffer
import Zygote:@ignore
import Functors:fmap
import Flux.Data:_getobs, _nobs
import Flux: @functor, gate, kaiming_normal, glorot_normal, zeros, trainable, Dense, glorot_uniform, σ, tanh, batched_vec, rpad
using Random
import DiffEqFlux:NeuralDELayer, basic_tgrad
using DiffEqBase
import DiffEqCallbacks: SavedValues, SavingCallback, PresetTimeCallback
using DiffEqSensitivity, DiffEqFlux
using DiffEqFlux: basic_tgrad, NeuralDELayer
using Requires


include("initializers.jl")
include("layers.jl")
include("interp.jl")
include("embedding.jl")
export CheapEmbedding


export limit_cycle, orthogonal_init
export ∂RNNCell, ∂GRUCell, ∂LSTMCell
export CubicSplineRegularGrid, LinearInterpolationRegularGrid, ConstantInterpolationRegularGrid, derivative, AbstractInterpolation
export CubicSpline

include("rnn_ode.jl")
export RNNODE, generate_adj_saving_callback

include("neural_cde.jl")
export NeuralCDE

function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
    include("interp_cuda.jl")
    end
end


end
