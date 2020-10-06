include("../layers.jl")
import Interpolations: degree, NoInterp, itpflag, tcollect, AbstractInterpolation
import DiffEqFlux:NeuralDELayer, basic_tgrad
import DifferentialEquations: DiscreteCallback
tdim = 2
xdim = 1
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
        if isnothing(p)
            p = _p
        end
        if isnothing(tspan)
            saveat = tspan[end]
        end
        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan), typeof(saveat),
            typeof(args),typeof(kwargs),typeof(xdim)}(
            model,p,re,tspan,saveat,args,kwargs,xdim)
    end
end
function derivative(x::A,t::Type{F}) where {F<:AbstractFloat, A<:Union{VecOrMat{<:AbstractFloat},CuArray{<:AbstractFloat}}}
    ẋ = diff(x,dims=tdim)
    Δt = 1 : size(ẋ,tdim)+1
    tstops = collect(t,Δt)
    condition(u,t,integrator) = t ∈ tstops
    affect!(integrator) = integrator.u.x[2]+= selectdim(ẋ,tdim, Int(integrator.t) )
    return PresetTimeCallback(tstops,affect!)
end

function (n::RNNODE)(x::A,p=n.p)
    where {A<:Union{VecOrMat{<:AbstractFloat},CuArray{<:AbstractFloat}}}

    function dudt_(u,p,t)
            ḣ = n.re(p)(u.x[1],u.x[2])
            ẋ = zeros(eltype(u))
            return ArrayPartition(ḣ,ẋ)
    end
    cb = derivative(x,eltype(n.tspan))
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,x,getfield(n,:tspan),p)
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    solve(prob,n.args...;saveat=n.saveat, sense=sense,callback =cb, n.kwargs...)
end

function (n::RNNODE)(x::T,p=n.p) where {T<:AbstractInterpolation}
    @assert ~isa(degree(filter(λ -> ~isa(λ, NoInterp), tcollect(itpflag, x))[1]), Constant)
    if isa(degree(filter(λ -> ~isa(λ, NoInterp), tcollect(itpflag, x))[1]), Linear)
        tstops = collect(eltype(n.tspan), itp.parentaxes[tdim])
    else
        tstops = nothing
    end
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


[c[:] for c in eachrow(x)]
o = LinearInterupolation.(STH,[1:length(x₀)])
