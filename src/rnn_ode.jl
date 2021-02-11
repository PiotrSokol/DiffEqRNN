"""
Redefining destructure/_restructure to handle additional arguments.
From https://github.com/FluxML/Flux.jl/pull/1353#issue-503431890
"""
# function destructure(m; cache = IdDict())
#   xs = Buffer([])
#   fmap(m) do x
#     if x isa AbstractArray
#       push!(xs, x)
#     else
#       cache[x] = x
#     end
#     return x
#   end
#   return vcat(vec.(copy(xs))...), p -> _restructure(m, p, cache = cache)
# end
#
# function _restructure(m, xs; cache = IdDict())
#   i = 0
#   fmap(m) do x
#     x isa AbstractArray || return cache[x]
#     x = reshape(xs[i.+(1:length(x))], size(x))
#     i += length(x)
#     return x
#   end
# end
# @adjoint function _restructure(m, xs; cache = IdDict())
#   _restructure(m, xs, cache = cache), dm -> (nothing,destructure(dm, cache = cache)[1])
# end
function _restructure(m, xs; cache = IdDict())
  i = 0
  fmap(m) do x
    x isa AbstractArray || return cache[x]
    x = reshape(xs[i.+(1:length(x))], size(x))
    i += length(x)
    return x
  end
end
function grabNrun(dm,cache)
    @show cache
    printstyled("Type of ∇ is : $(typeof(dm)) and its size is $(size(dm))", color=:red)
    return (nothing,destructure(dm,  cache = cache)[1])
    # dm -> (nothing,destructure(dm,  cache = cache)[1])
end
@adjoint function _restructure(m, xs; cache = IdDict())
  _restructure(m, xs, cache = cache), dm->grabNrun(dm,cache)
end

"""
    destructure(m)
Flatten a model's parameters into a single weight vector.
    julia> m = Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
    Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
    julia> θ, re = destructure(m);
    julia> θ
    67-element Array{Float32,1}:
    -0.1407104
    ...
The second return value `re` allows you to reconstruct the original network after making
modifications to the weight vector (for example, with a hypernetwork).
    julia> re(θ .* 2)
    Chain(Dense(10, 5, σ), Dense(5, 2), softmax)
"""
function destructure(m; cache = IdDict())
  xs = Buffer([])
  fmap(m) do x
    x isa AbstractArray ? push!(xs, x) : (cache[x] = x)
    return x
  end
  return vcat(vec.(copy(xs))...), p -> _restructure(m, p, cache = cache)
end
struct RNNODE{M<:AbstractRNNDELayer,P,RE,T,A,K,F,S} <: NeuralDELayer
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

    function RNNODE(model,tspan, args...;p = nothing, preprocess=permutedims,
        sense = InterpolatingAdjoint(autojacvec=ZygoteVJP()), append_input=false, kwargs...)

        _p,re = Flux.destructure(model)
        nhidden = size(model.Wᵣ,2)
        nin = size(model.Wᵢ,2)
        if isnothing(p)
            p = _p
        end

        u₀ = similar(p,nhidden)
        u₀ .= 2rand(eltype(u₀), size(u₀)).-1

        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(args),typeof(kwargs),typeof(preprocess),
            append_input}(
            model,u₀,p,re,tspan,args,kwargs,sense,nin,
            nhidden, preprocess)
    end
    function RNNODE(model::∂LSTMCell,tspan, args...;p = nothing,  preprocess=permutedims, sense = InterpolatingAdjoint(autojacvec=ZygoteVJP()), append_input=false, kwargs...)

        _p,re = Flux.destructure(model)
        nhidden = size(model.Wᵢ,1)÷2
        nin = size(model.Wᵢ,2)
        if isnothing(p)
            p = _p
        end

        u₀ = similar(p,2nhidden)
        u₀ .= 2rand(eltype(u₀), size(u₀)).-1

        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(args),typeof(kwargs),typeof(preprocess),append_input}(
            model,p,re,tspan,args,kwargs,sense,nin,
            nhidden, preprocess)
    end
end

# function Base.getproperty(n::RNNODE{<:AbstractRNNDELayer}, sym::Symbol)
#   if sym === :u₀
#     return getfield(n.model, sym)
#   else
#     return getfield(n, sym)
#   end
# end

function Base.show(io::IO, l::RNNODE)
  print(io, "RNNODE(", l.in, ", ", l.hidden)
  print(io, ")")
end

"""
  Returns default u₀ repeated to match the batch size inferred from the interpolation.
  Assumes that the last dimension of n.preprocess ∘ X(t) represents batch size.
"""
function get_u₀(X::AbstractInterpolation, n::RNNODE)
  inferred_batchsize = @ignore size(n.preprocess(X(minimum(X.t))))[end]
  u₀ = repeat(n.u₀, 1, inferred_batchsize)
end

function get_u₀(X::AbstractMatrix{<:AbstractFloat}, n::RNNODE)
  inferred_batchsize = @ignore size(n.preprocess(X[1,:]))[end]
  u₀ = repeat(n.u₀, 1, inferred_batchsize)
end

function (n::RNNODE{M,P,RE,T,A,K,F,S})(X::IT, p::P=n.p, u₀::UT=get_u₀(X,n)) where{M,P,RE,T,A,K,F,S, IT<:Union{CubicSpline,CubicSplineRegularGrid},
  UT<:AbstractArray{<:AbstractFloat}}

    # dudt_= let re=n.re, x=X, f::F = n.preprocess
    #   (u,p,t) -> re(p)(u,f(X(t)))
    # end
    dudt_= let re=n.re, x=X
      (u,p,t) -> re(p)(u,permutedims(X(t)))
    end
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    solve(prob,n.args...;sense=n.sense, n.kwargs...)
end
# """
#   Dispatches a special method for an RNN ODE where the input is appended as a state variable to the ODE solver.

#   Motivate by practical observations that the ODE solver fails to adapt to the smoothness of the inhomogeneous input. In principle ought to be resolved using `d_dicsontinuities` judiciously.

#   Offers a potential trade-off between the number of solver calls (and hence also overall peak memory usage) and memory usage due to the extended state space.
# """
# const ExtendedStateSpaceRNNODE = RNNODE{<:AbstractRNNDELayer, <:Any,<:Any,<:Any,<:Any,<:Any,true}

# function (n::ExtendedStateSpaceRNNODE)(X::IT, p::PT=n.p, u₀::UT=get_u₀(X,n)) where {IT<:Union{CubicSpline,CubicSplineRegularGrid},
#         UT<:AbstractArray{<:AbstractFloat},
#         PT<:AbstractVector{<:AbstractFloat}}
#     x = nograd(X, f=permutedims!)
#     nchannels = @ignore infer_batchsizes(x) ÷ size(x.interpolant.u, 1)
#     tspan = getfield(n,:tspan)
#     _u₀ = vcat(u₀, reshape(X(tspan[1]), nchannels, :))
#     dudt_(u,p,t) = let re=n.re, x=x, nchannels=nchannels, f=n.preprocess
#       (u,p,t) -> begin
#         ũ = @views u[1:end-nchannels,:]
#         xₜ = @views u[end-nchannels+1,:]
#         du = re(p)(ũ, f(xₜ))
#         vcat(du, derivative(x,t) )
#       end
#     end
#     ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
#     prob = ODEProblem{false}(ff,_u₀,tspan,p)
#     solve(prob,n.args...;sense=n.sense, n.kwargs...)
# end
# """
# Because of the low-order smoothness of LinearInterpolation, we force the solver to restart at new each data
# """
function (n::RNNODE{M,P,RE,T,A,K,F,S})(X::IT, p::P=n.p, u₀::UT=get_u₀(X,n)) where {M,P,RE,T,A,K,F,S, IT<:Union{LinearInterpolation,LinearInterpolationRegularGrid},
  UT<:AbstractArray{<:AbstractFloat}}

    tstops = eltype(n.u₀).(collect(X.t))

    dudt_= let re=n.re, x=X, f::F=n.preprocess
      (u,p,t) -> re(p)(u, f(X(t)))
    end

    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    solve(prob,n.args...;sense=n.sense, tstops=tstops, n.kwargs...)
end

function (n::RNNODE{M,P,RE,T,A,K,F,S})(X::ConstantInterpolation, p::P=n.p, u₀::UT=get_u₀(X,n)) where {M,P,RE,T,A,K,F,S, UT<:AbstractArray{<:AbstractFloat}}
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

    dudt_ = let re=n.re, f::F=n.preprocess
      (u,p,t) -> begin
        ũ = @views u[1:end-nchannels,:]
        xₜ = @views u[end-nchannels+1,:]
        du = n.re(p)(ũ,n.preprocess(xₜ))
        return vcat(du, reshape(zero(xₜ), nchannels, :))
      end
    end
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,_u₀,tspan,p)
    solve(prob,n.args...;callback = cb, sense=n.sense, n.kwargs...)
end

"""
  Dispatch for RNN with an Array of inputs.
  Treats the system of equations as an impulsive ODE perturbed by the differenced array evaluated on an integer grid with ∆t = 1

  size(X,1) is (channels) × (batch size)
  size(X,2) is time
"""
# function (n::RNNODE)(X::IT, u₀::UT=get_u₀(X,n), p::PT=n.p) where  {IT<:AbstractArray{<:AbstractFloat}, UT<:AbstractArray{<:AbstractFloat}, PT<:AbstractVector{<:AbstractFloat}}

#     ∂x = ignore() do
#       hcat(X[:,1],diff(X, dims=2))
#     end
#     tstops = 1:size(∂x,2)
#     tspan = getfield(n,:tspan)

#     nchannels, cb = ignore() do
#       inferred_batchsize =n.preprocess(∂x[1,:])[end]
#       nchannels = inferred_batchsize ÷ size(∂x,1)
#       affect!(integrator) = integrator.u[end-nchannels+1, :] += view(∂x, Int(integrator.t),:)
#       cb = PresetTimeCallback(tstops, affect!, save_positions=(false,false))
#       return nchannels, cb
#     end

#     _u₀ = vcat(u₀,reshape(∂x(1,:), nchannels, :))

#     dudt_(u,p,t) = begin
#       ũ = @views u[1:end-nchannels,:]
#       xₜ = @views u[end-nchannels+1,:]
#       du = n.re(p)(ũ,n.preprocess(xₜ))
#       return vcat(du, reshape(zero(xₜ), nchannels, :))
#     end
#     ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
#     prob = ODEProblem{false}(ff,_u₀,tspan,p)
# end
"""
RNNODE with no input x defines an IVP for a homogenous system
"""
function (n::RNNODE{M,P,RE,T,A,K,F,S})(u₀::UT, p::P=n.p) where {M,P,RE,T,A,K,F,S,UT<:AbstractArray{<:AbstractFloat}}
    dudt_(u,p,t) = n.re(p)(u)
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
