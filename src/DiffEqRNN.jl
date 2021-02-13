module DiffEqRNN

using DataInterpolations
import DataInterpolations: AbstractInterpolation, munge_data, derivative
import LinearAlgebra:Tridiagonal, qr, Diagonal

import Zygote:ignore, @ignore, Buffer, @adjoint
import Functors:fmap
import Flux.Data:_getobs, _nobs
import Flux: @functor, gate, kaiming_normal, glorot_normal, zeros, trainable, Dense, glorot_uniform, σ, tanh, batched_vec, rpad
using Flux
# import Flux:_restructure,destructure


using DiffEqBase,DiffEqSensitivity, DiffEqFlux
import DiffEqFlux:NeuralDELayer, basic_tgrad
import DiffEqCallbacks: SavedValues, SavingCallback, PresetTimeCallback

using Random
using Requires


include("initializers.jl")
# include("layers.jl")
# export ∂RNNCell, ∂GRUCell, ∂LSTMCell
include("fast_layers.jl")
include("interp.jl")
include("nograd.jl")

export limit_cycle, orthogonal_init
export Fast∂RNNCell, Fast∂GRUCell, Fast∂LSTMCell
export CubicSplineRegularGrid, LinearInterpolationRegularGrid, ConstantInterpolationRegularGrid, derivative, AbstractInterpolation
export CubicSpline

# include("rnn_ode.jl")
include("fast_rnn_ode.jl")
export RNNODE, generate_adj_saving_callback

include("neural_cde.jl")
export NeuralCDE

function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include("interp_cuda.jl")
    end
    @require LearnBase="7f8f8fb0-2700-5f03-b4bd-41f8cfc144b6" begin
        using .LearnBase
        LearnBase.getobs(data::ConstantInterpolation{<:AbstractMatrix}, i) = ConstantInterpolation{true}(data.u[i,:],data.t,data.dir)
        LearnBase.getobs(data::LinearInterpolationRegularGrid{<:AbstractMatrix}, i) = LinearInterpolation{true}(data.u[i,:],data.t)
        LearnBase.getobs(data::CubicSplineRegularGrid{<:AbstractMatrix}, i) = CubicSplineRegularGrid{true}(data.u[i,:], data.t.start, data.t.stop, data.t.step, data.z[i,:])
        LearnBase.getobs(data::CubicSpline{<:AbstractMatrix}, i) = CubicSpline{true}(data.u[i,:], data.t, data.h, data.z[i,:])
        LearnBase.nobs(n::T) where {T<:Union{ConstantInterpolation,LinearInterpolationRegularGrid,CubicSplineRegularGrid,CubicSpline}} =  size(n.u,1)
    end
end


end
