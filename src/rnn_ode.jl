include("/Users/piotrsokol/Documents/RNNODE.jl/src/interp.jl")
include("/Users/piotrsokol/Documents/RNNODE.jl/src/layers.jl")
import DiffEqFlux:NeuralDELayer, basic_tgrad
using DifferentialEquations, DiffEqCallbacks
import OrdinaryDiffEq: ODEFunction, ODEProblem, solve
import DiffEqSensitivity: InterpolatingAdjoint, ZygoteVJP

struct RNNODE{M<:AbstractRNNDELayer,P,RE,T,A,K,VM,I} <: NeuralDELayer
    model::M
    p::P
    re::RE
    tspan::T
    args::A
    kwargs::K
    u₀::VM
    in::I
    hidden::I
    preprocess

    function RNNODE(model,tspan, args...;p = nothing, u₀ = nothing, preprocess= identity, kwargs...)
        _p,re = Flux.destructure(model)
        nhidden = size(model.Wᵣ)[2]
        nin = size(model.Wᵢ)[2]
        if isnothing(p)
            p = _p
        end
        if isnothing(u₀)
            u₀ = 2rand(eltype(model.Wᵣ), nhidden ,1).-1  # code adapted to sigmoidal (tanh) RNNs -> for this reason u₀ ~ Unif over the [-1, 1] hypercube
        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(args),typeof(kwargs),typeof(u₀),
            typeof(nin)}(
            model,p,re,tspan,args,kwargs,u₀,nin,
            nhidden, preprocess)
        end
    end
    function RNNODE(model::∂LSTMCell,tspan, args...;p = nothing, u₀ = nothing, preprocess=identity, kwargs...)
        _p,re = Flux.destructure(model)
        nhidden = size(model.Wᵢ)[1]÷2
        nin = size(model.Wᵢ)[2]
        if isnothing(p)
            p = _p
        end
        if isnothing(u₀)
            u₀ = 2rand(eltype(model.Wᵣ), nhidden ,1).-1  # code adapted to sigmoidal (tanh) RNNs -> for this reason u₀ ~ Unif over the [-1, 1] hypercube
        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(args),typeof(kwargs),typeof(u₀),
            typeof(nin)}(
            model,p,re,tspan,args,kwargs,u₀,nin,
            nhidden, preprocess)
        end
    end
end

function (n::RNNODE)(X::T; u₀=nothing, p=n.p) where {T<:Union{CubicSpline,CubicSplineFixedGrid}}
    x = nograd(X, f=n.preprocess)
    if isnothing(u₀)
        u₀ = repeat(n.u₀, 1, batchsize(x))
    end
    dudt_(u,p,t) = n.re(p)(u,x(t))
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    solve(prob,n.args...;sense=sense, n.kwargs...)
end
"""
Because of the low-order smoothness of LinearInterpolation and ConstantInterpolation, we force the solver to restart at new each data
"""
function (n::RNNODE)(X::T; u₀=nothing, p=n.p) where {T<:Union{LinearInterpolation,ConstantInterpolation}}
    x = nograd(X, f=n.preprocess)
    tstops = eltype(n.u₀).(collect(cpt.t))
    if isnothing(u₀)
        u₀ = repeat(n.u₀, 1, batchsize(x))
    end
    dudt_(u,p,t) = n.re(p)(u,x(t))
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    solve(prob,n.args...;sense=sense, tstops=tstops, n.kwargs...)
end
"""
RNNODE with no input x defines an IVP for a homogenous system
"""
function (n::RNNODE)(u₀; p=n.p)
    dudt_(u,p,t) = n.re(p)(u, zeros(eltype(u₀), n.in, 1) )
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    solve(prob,n.args...;sense=sense, n.kwargs...)
end
"""
Helper code for saving adjoint variables and doing "gradient clipping"
"""
function generate_adj_saving_callback(rnn::AbstractRNNDELayer, saveat, bs::Int;hidden::Int = size(rnn.Wᵣ)[2],f::Function = identity)

    saved_values = SavedValues(eltype(saveat), Array)
    function save_func(u,t,integrator)
        uˌ = u[1:bs*hidden]
        uˌ= reshape(uˌ,hidden,bs)
        return hcat(f.(eachcol(uˌ))...)
    end
    cb = SavingCallback(save_func, saved_values; saveat=saveat,tdir=-1)
    return cb, saved_values
end
