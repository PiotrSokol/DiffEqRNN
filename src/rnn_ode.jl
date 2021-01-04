"""
Redefining destructure/_restructure to handle additional arguments.
From https://github.com/FluxML/Flux.jl/pull/1353#issue-503431890
"""
function destructure(m; cache = IdDict())
  xs = Buffer([])
  fmap(m) do x
    if x isa AbstractArray
      push!(xs, x)
    else
      cache[x] = x
    end
    return x
  end
  return vcat(vec.(copy(xs))...), p -> _restructure(m, p, cache = cache)
end

function _restructure(m, xs; cache = IdDict())
  i = 0
  fmap(m) do x
    x isa AbstractArray || return cache[x]
    x = reshape(xs[i.+(1:length(x))], size(x))
    i += length(x)
    return x
  end
end

struct RNNODE{M<:AbstractRNNDELayer,P,RE,T,A,K,I} <: NeuralDELayer
    model::M
    p::P
    re::RE
    tspan::T
    args::A
    kwargs::K
    sense
    in::I
    hidden::I
    preprocess

    function RNNODE(model,tspan, args...;p = nothing, preprocess = identity,
        sense = InterpolatingAdjoint(autojacvec=ZygoteVJP()), kwargs...)
        _p,re = destructure(model)
        nhidden = size(model.Wᵣ,2)
        nin = size(model.Wᵢ,2)
        if isnothing(p)
            p = _p
        end

        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(args),typeof(kwargs),
            typeof(nin)}(
            model,p,re,tspan,args,kwargs,sense,nin,
            nhidden, preprocess)
    end
    function RNNODE(model::∂LSTMCell,tspan, args...;p = nothing, preprocess=identity, sense = InterpolatingAdjoint(autojacvec=ZygoteVJP()), kwargs...)
        _p,re = destructure(model)
        nhidden = size(model.Wᵢ,1)÷2
        nin = size(model.Wᵢ,2)
        if isnothing(p)
            p = _p
        end

        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(args),typeof(kwargs),
            typeof(nin)}(
            model,p,re,tspan,args,kwargs,sense,nin,
            nhidden, preprocess)
    end
end

function Base.getproperty(n::RNNODE{<:AbstractRNNDELayer}, sym::Symbol)
  if sym === :u₀
    return getfield(n.model, sym)
  else
    return getfield(n, sym)
  end
end

function Base.show(io::IO, l::RNNODE)
  print(io, "RNNODE(", l.in, ", ", l.hidden)
  print(io, ")")
end

function (n::RNNODE)(X::T; u₀=nothing, p=n.p) where {T<:Union{CubicSpline,CubicSplineRegularGrid}}
    x = nograd(X, f=n.preprocess)
    if isnothing(u₀)
        u₀ = repeat(n.u₀, 1, get_batchsize(x)) |> deepcopy
    end
    dudt_(u,p,t) = n.re(p)(u,x(t))
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    solve(prob,n.args...;sense=n.sense, n.kwargs...)
end
"""
Because of the low-order smoothness of LinearInterpolation and ConstantInterpolation, we force the solver to restart at new each data
"""
function (n::RNNODE)(X::T; u₀=nothing, p=n.p) where {T<:Union{LinearInterpolation,LinearInterpolationRegularGrid,ConstantInterpolation}}
    x = nograd(X, f=n.preprocess)
    tstops = eltype(n.u₀).(collect(X.t))
    if isnothing(u₀)
        u₀ = repeat(n.u₀, 1, get_batchsize(x)) |> deepcopy
    end
    dudt_(u,p,t) = n.re(p)(u,x(t))
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    solve(prob,n.args...;sense=n.sense, tstops=tstops, n.kwargs...)
end
"""
RNNODE with no input x defines an IVP for a homogenous system
"""
function (n::RNNODE)(u₀::AbstractVecOrMat{<:Number}; p=n.p)
    dudt_(u,p,t) = n.re(p)(u, zeros(eltype(u₀), n.in, 1) )
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    solve(prob,n.args...;sense=n.sense, n.kwargs...)
end
"""
Helper code for saving adjoint variables
"""
function generate_adj_saving_callback(rnn::AbstractRNNDELayer, saveat, bs::Int;hidden::Int = rnn.hidden,f::Function = identity)

    saved_values = SavedValues(eltype(saveat), Array)
    function save_func(u,t,integrator)
        uˌ = u[1:bs*hidden]
        uˌ= reshape(uˌ,hidden,bs)
        return hcat(f.(eachcol(uˌ))...)
    end
    cb = SavingCallback(save_func, saved_values; saveat=saveat,tdir=-1)
    return cb, saved_values
end
