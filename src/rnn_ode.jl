basic_tgrad(u,p,t) = zero(u)
struct RNNODE{M,P,RE,T,A,K,F,S} <: NeuralDELayer
    model::M
    u₀::P
    p::P
    re::RE
    tspan::T
    args::A
    kwargs::K
    sense
    in::Integer
    hidden::Integer
    preprocess::F
    #extendedstates::S

    function RNNODE(model::FastRNNLayer,tspan, args...;p=initial_params(model), preprocess=permutedims, sense = InterpolatingAdjoint(autojacvec=ZygoteVJP()), append_input=false, kwargs...)

        re = nothing
        nhidden = model.out
        nin = model.out
        u₀ = model.u₀
      new{typeof(model),typeof(p),typeof(re),
          typeof(tspan),typeof(args),typeof(kwargs),typeof(preprocess),
          append_input}(
          model,u₀,p,re,tspan,args,kwargs,sense,nin,
          nhidden, preprocess)
    end
end

function Base.getproperty(n::RNNODE, sym::Symbol)
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

"""
  Returns default u₀ repeated to match the batch size inferred from the interpolation.
  Assumes that the last dimension of n.preprocess ∘ X(t) represents batch size.
"""
function get_u₀(X::AbstractInterpolation, n::NeuralDELayer)
  inferred_batchsize = @ignore size(n.preprocess(X(minimum(X.t))))[end]
  u₀ = repeat(n.u₀, 1, inferred_batchsize)
end

function get_u₀(X::AbstractMatrix{<:AbstractFloat}, n::NeuralDELayer)
  inferred_batchsize = @ignore size(n.preprocess(X[1,:]))[end]
  u₀ = repeat(n.u₀, 1, inferred_batchsize)
end
##
function (n::RNNODE{M,P,RE,T,A,K,F,S})(X::IT, p::P=n.p, u₀::UT=get_u₀(X,n)) where{M<:FastRNNLayer,P,RE,T,A,K,F,S, IT<:Union{CubicSpline,CubicSplineRegularGrid},
  UT<:AbstractArray{<:AbstractFloat}}

    dudt_= let x=X, model = n.model, f::F=n.preprocess
      (u,p,t) -> model(u,f(x(t)),p)
    end

    ff = ODEFunction{false}(dudt_,tgrad=(u,p,t)->zero(eltype(T)))
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    solve(prob,n.args...;sense=n.sense, n.kwargs...)
end

function (n::RNNODE{M,P,RE,T,A,K,F,S})(X::IT, p::P=n.p, u₀::UT=get_u₀(X,n)) where {M<:FastRNNLayer,P,RE,T,A,K,F,S, IT<:Union{LinearInterpolation,LinearInterpolationRegularGrid},
  UT<:AbstractArray{<:AbstractFloat}}

    tstops = eltype(u₀).(collect(X.t))

    dudt_= let x=X, model = n.model, f::F=n.preprocess
      (u,p,t) -> model(u,f(x(t)),p)
    end

    ff = ODEFunction{false}(dudt_,tgrad=(u,p,t)->zero(eltype(T)))
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    solve(prob,n.args...;sense=n.sense, tstops=tstops, n.kwargs...)
end

function (n::RNNODE{M,P,RE,T,A,K,F,S})(X::ConstantInterpolation, p::P=n.p, u₀::UT=get_u₀(X,n)) where {M<:FastRNNLayer,P,RE,T,A,K,F,S, UT<:AbstractArray{<:AbstractFloat}}

    x = nograd(X, f=identity)
    ∂x = ignore() do
      ∂x = nograd(ConstantInterpolation{true}(hcat(X.u[:,1],diff(X.u, dims=2)), X.t,X.dir), f=identity)
    end

    tstops = eltype(n.u₀).(collect(X.t))
    tspan = getfield(n,:tspan)

    nchannels, cb = ignore() do
      nchannels = infer_batchsizes(∂x) ÷ size(∂x.interpolant.u,1)
      affect!(integrator) = integrator.u[end-nchannels+1, :] += ∂x(integrator.t)
      cb = PresetTimeCallback(setdiff(tstops,tspan), affect!, save_positions=(false,false))
      return nchannels, cb
    end
    _u₀ = vcat(u₀,reshape(x(tspan[1]), nchannels, :))

    dudt_ = let model=n.model, f::F=n.preprocess
      (u,p,t) -> begin
        ũ = @views u[1:end-nchannels,:]
        xₜ = @views u[end-nchannels+1,:]
        du = model(ũ,f(xₜ),p)
        return vcat(du, reshape(zero(xₜ), nchannels, :))
      end
    end
    ff = ODEFunction{false}(dudt_,tgrad=(u,p,t)->zero(eltype(T)))
    prob = ODEProblem{false}(ff,_u₀,tspan,p)
    solve(prob,n.args...;callback = cb, sense=n.sense, n.kwargs...)
end

function (n::RNNODE{M,P,RE,T,A,K,F,S})(u₀::UT, p::P=n.p) where {M<:FastRNNLayer,P,RE,T,A,K,F,S,UT<:AbstractArray{<:AbstractFloat}}
    dudt_(u,p,t) = n.model(u,p)
    ff = ODEFunction{false}(dudt_,tgrad=(u,p,t)->zero(eltype(T)))
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
        return reduce( hcat, f.(eachcol(uˌ)))
    end
    cb = SavingCallback(save_func, saved_values; saveat=saveat,tdir=-1)
    return cb, saved_values
end
