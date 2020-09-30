include("../layers.jl")
import Interpolations: degree, NoInterp, itpflag, tcollect, AbstractInterpolation
import DiffEqFlux:NeuralDELayer, basic_tgrad
import DifferentialEquations: DiscreteCallback

struct RNNODE{M<:AbstractRNNDELayer,P,RE,T,A,K,S,I} <: NeuralDELayer
    model::M
    p::P
    re::RE
    tspan::T
    saveat::T
    args::A
    kwargs::K
    xdim::I


    function RNNODE(model,tspan, xdim, args...;p = nothing, saveat = nothing, kwargs...)
        _p,re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        if saveat === nothing
            saveat = tspan[end]
        end
        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan), typeof(saveat),
            typeof(args),typeof(kwargs),typeof(xdim)}(
            model,p,re,tspan,saveat,args,kwargs,xdim)
    end
end
function derivative(x::A,t::Type{F}) where {F<:AbstractFloat, A<:Union{VecOrMat{<:AbstractFloat},CuArray{<:AbstractFloat}}}
    ẋ = diff(x,dims=TDims)
    Δt = 1 : size(ẋ,TDims)+1
    tstops = collect(t,Δt)
    condition(u,t,integrator) = t ∈ tstops
    affect!(integrator) = integrator.u.x[2]+= selectdim(ẋ,TDims, Int(integrator.t) )
    # TODO: replace DiscreteCallback with PresetTimeCallback
    # return DiscreteCallback(condition,affect!), tstops
    return PresetTimeCallback(tstops,affect!)
end
function (n::RNNODE)(x::A,p=n.p) where {A<:Union{VecOrMat{<:AbstractFloat},CuArray{<:AbstractFloat}}}
    # This is the version that deals with non-interpolated data:
    # it needs to spit out time indices for tstops, in appropriate numeric format,
    # taken from tspan

    function dudt_(u,p,t)
            ḣ = n.re(p)(u.x[1],u.x[2])
            ẋ = zeros(eltype(u))
            return ArrayPartition(ḣ,ẋ)
    end
    callback, tstops = derivative(x,eltype(n.tspan))
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,x,getfield(n,:tspan),p)
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    solve(prob,n.args...;saveat=n.saveat, sense=sense,n.kwargs...)
end

function (n::RNNODE)(x::T,p=n.p) where {T<:AbstractInterpolation}
    # This is the version that deals with non-interpolated data:
    # it needs to spit out time indices for tstops, in appropriate numeric format,
    # taken from tspan
    function dudt_(u,p,t)
            ḣ = n.re(p)(u.x[1],u.x[2])
            ẋ = zeros(eltype(u))
            return ArrayPartition(ḣ,ẋ)
    end
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,x,getfield(n,:tspan),p)
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    solve(prob,n.args...;sense=sense,tstops, n.kwargs...)
end
