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

struct RNNODE{M<:AbstractRNNDELayer,P,RE,T,A,K,I,S} <: NeuralDELayer
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
    #extendedstates::S

    function RNNODE(model,tspan, args...;p = nothing, preprocess = identity,
        sense = InterpolatingAdjoint(autojacvec=ZygoteVJP()), append_input=false, kwargs...)
        _p,re = destructure(model)
        nhidden = size(model.Wᵣ,2)
        nin = size(model.Wᵢ,2)
        if isnothing(p)
            p = _p
        end

        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(args),typeof(kwargs),
            typeof(nin), append_input}(
            model,p,re,tspan,args,kwargs,sense,nin,
            nhidden, preprocess)
    end
    function RNNODE(model::∂LSTMCell,tspan, args...;p = nothing, preprocess=identity, sense = InterpolatingAdjoint(autojacvec=ZygoteVJP()), append_input=false, kwargs...)
        _p,re = destructure(model)
        nhidden = size(model.Wᵢ,1)÷2
        nin = size(model.Wᵢ,2)
        if isnothing(p)
            p = _p
        end

        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(args),typeof(kwargs),
            typeof(nin), append_input}(
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


function (n::RNNODE)(X::T; u₀=nothing, p=n.p) where{T<:Union{CubicSpline,CubicSplineRegularGrid}}
    x = nograd(X, f=n.preprocess)
    if isnothing(u₀)
        u₀ = repeat(n.u₀, 1, infer_batchsizes(x))
    end
    dudt_(u,p,t) = n.re(p)(u,x(t))
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    solve(prob,n.args...;sense=n.sense, n.kwargs...)
end
"""
  Dispatches a special method for an RNN ODE where the input is appended as a state variable to the ODE solver.
  
  Motivate by practical observations that the ODE solver fails to adapt to the smoothness of the inhomogeneous input. In principle ought to be resolved using `d_dicsontinuities` judiciously.

  Offers a potential trade-off between the number of solver calls (and hence also overall peak memory usage) and memory usage due to the extended state space.
"""
const ExtendedStateSpaceRNNODE = RNNODE{<:AbstractRNNDELayer, <:Any,<:Any,<:Any,<:Any,<:Any,<:Any,true}

function (n::ExtendedStateSpaceRNNODE)(X::T; u₀=nothing, p=n.p)  where{T<:Union{CubicSpline,CubicSplineRegularGrid}}
    x = nograd(X, f=n.preprocess)
    if isnothing(u₀)
        u₀ = repeat(n.u₀, 1, infer_batchsizes(x))
    end
    nchannels = @ignore infer_batchsizes(x) ÷ size(x.interpolant.u,1)
    tspan = getfield(n,:tspan)
    _u₀ = vcat(u₀, reshape(x(tspan[1]), nchannels, :))
    dudt_(u,p,t) = begin 
      ũ = u[1:end-nchannels,:]
      xₜ = u[end-nchannels+1,:]
      du = n.re(p)(ũ,n.preprocess(xₜ))
      return vcat(du, derivative(x,t) )
    end
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,_u₀,tspan,p)
    solve(prob,n.args...;sense=n.sense, n.kwargs...)
end


"""
Because of the low-order smoothness of LinearInterpolation, we force the solver to restart at new each data
"""
function (n::RNNODE)(X::T; u₀=nothing, p=n.p) where {T<:Union{LinearInterpolation,LinearInterpolationRegularGrid}}
    x = nograd(X, f=n.preprocess)
    tstops = eltype(n.u₀).(collect(X.t))
    if isnothing(u₀)
        u₀ = repeat(n.u₀, 1, infer_batchsizes(x))
    end
    dudt_(u,p,t) = n.re(p)(u,x(t))
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    solve(prob,n.args...;sense=n.sense, tstops=tstops, n.kwargs...)
end
function (n::RNNODE)(X::ConstantInterpolation; u₀=nothing, p=n.p)
    
    x = nograd(X, f=identity)
    ∂x = ignore() do
      ∂x = nograd(ConstantInterpolation{true}(hcat(X.u[:,1],diff(X.u, dims=2)), X.t,X.dir), f=identity)
    end
    tstops = eltype(n.u₀).(collect(X.t))
    tspan = getfield(n,:tspan)
    if isnothing(u₀)
        u₀ = repeat(n.u₀, 1, infer_batchsizes(∂x))
    end
    nchannels, cb = ignore() do
      nchannels = infer_batchsizes(∂x) ÷ size(∂x.interpolant.u,1)
      affect!(integrator) = integrator.u[end-nchannels+1, :] += ∂x(integrator.t)
      cb = PresetTimeCallback(setdiff(tstops,tspan), affect!, save_positions=(false,false))
      return nchannels, cb
    end
    _u₀ = vcat(u₀,reshape(x(tspan[1]), nchannels, :))

    dudt_(u,p,t) = begin 
      ũ = u[1:end-nchannels,:]
      xₜ = u[end-nchannels+1,:]
      du = n.re(p)(ũ,n.preprocess(xₜ))
      return vcat(du, reshape(zero(xₜ), nchannels, :))
    end
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,_u₀,tspan,p)
    solve(prob,n.args...;callback = cb, sense=n.sense, n.kwargs...)
end
"""
RNNODE with no input x defines an IVP for a homogenous system
"""
function (n::RNNODE)(u₀::AbstractVecOrMat{<:Number}; p=n.p)
    x = ignore() do
      fill!(similar(n.u₀, n.in, size(u₀,2),), zero(eltype(u₀)))
    end
    dudt_(u,p,t) = n.re(p)(u, x )
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    solve(prob,n.args...;sense=n.sense, n.kwargs...)
end
"""
Helper code for saving adjoint variables
"""
function generate_adj_saving_callback(rnn::NeuralDELayer, saveat, bs::Int; hidden::Int = rnn.hidden,f::Function = identity)

    saved_values = SavedValues(eltype(saveat), Array)
    function save_func(u,t,integrator)
        uˌ = u[1:bs*hidden]
        uˌ= reshape(uˌ,hidden,bs)
        return hcat(f.(eachcol(uˌ))...)
    end
    cb = SavingCallback(save_func, saved_values; saveat=saveat,tdir=-1)
    return cb, saved_values
end
