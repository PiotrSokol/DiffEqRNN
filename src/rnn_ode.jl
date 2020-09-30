include("../layers.jl")
import Interpolations: degree, NoInterp, itpflag, tcollect, Constant, Linear, Quadratic, Cubic
import DiffEqFlux:NeuralDELayer, basic_tgrad

struct RNNODE{M<:AbstractRNNDELayer,P,RE,T,A,K,S,I} <: NeuralDELayer
    model::M
    p::P
    re::RE
    tspan::T
    args::A
    kwargs::K
    interp::S
    xdim::I


    function NeuralODE(model,tspan,args...;p = nothing,interp, kwargs...)
        _p,re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(args),typeof(kwargs)}(
            model,p,re,tspan,args,kwargs)
    end
end

function (n::NeuralODE)(x,p=n.p)
    if ~isa(itp, Interpolations.AbstractInterpolation)
    elseif isa(degree(filter(x -> ~isa(x, NoInterp), tcollect(itpflag, itp))[1]), Constant)
    elseif isa(degree(filter(x -> ~isa(x, NoInterp), tcollect(itpflag, itp))[1]), Linear)
    elseif isa(degree(filter(x -> ~isa(x, NoInterp), tcollect(itpflag, itp))[1]), Union{Quadratic,Cubic})
    end

    dudt_(u,p,t) = n.re(p)(u)
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,x,getfield(n,:tspan),p)
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    solve(prob,n.args...;sense=sense,n.kwargs...)
end
end
