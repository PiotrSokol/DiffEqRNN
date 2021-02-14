
struct NeuralCDE{M,P,RE,T,A,K} <: NeuralDELayer
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
    preprocess
    function NeuralCDE(model,tspan, nin::Int, nhidden::Int, args...;p=nothing, preprocess=nothing, u₀=nothing, sense=InterpolatingAdjoint(autojacvec=ZygoteVJP()), kwargs...)
        _p,re = Flux.destructure(model)
        if isnothing(p)
            p = _p
        end
        if isnothing(preprocess)
            preprocess = x-> reshape(x,nin,:)
        end
        if isnothing(u₀)
            u₀=Random.randn!(similar(p,nhidden))./eltype(p)(sqrt(nhidden))
        end

        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(args),typeof(kwargs)}(
            model,u₀,p,re,tspan,args,kwargs,sense,nin,
            nhidden, preprocess)
    end
    function NeuralCDE(model::M,tspan, nin::Int, nhidden::Int, args...;p=initial_params(model), preprocess=nothing, u₀=nothing, sense=InterpolatingAdjoint(autojacvec=ZygoteVJP()), kwargs...) where {M<:Union{FastLayer,FastChain}}
        
        re = nothing 
        if isnothing(preprocess)
            preprocess = x-> reshape(x,nin,:)
        end
        if isnothing(u₀)
            u₀=Random.randn!(similar(p,nhidden))./eltype(p)(sqrt(nhidden))
        end

        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(args),typeof(kwargs)}(
            model,u₀,p,re,tspan,args,kwargs,sense,nin,
            nhidden, preprocess)
    end
end
function dudt_CDE(u,p,t,X,re,in,hidden, preprocess)
        u = permutedims(reshape( re(p)(u), in, hidden,:), [2,1,3])
        dX = derivative(X,t) |> preprocess
        batched_vec(u,dX)
end

function Fast_dudt_CDE(u,p,t,X,model,in,hidden, preprocess)
        u = permutedims(reshape( model(u,p), in, hidden,:), [2,1,3])
        dX = derivative(X,t) |> preprocess
        batched_vec(u,dX)
end

function (n::NeuralCDE)(X::T, p=n.p, u₀=get_u₀(X,n)) where {T<:Union{CubicSpline,CubicSplineRegularGrid}}
    # x = nograd(X, f=n.preprocess)
    if isnothing(u₀)
      inferred_batchsize = ignore() do
        size(n.preprocess(X(minimum(X.t))))[end]
      end
        u₀ = repeat(n.u₀, 1, inferred_batchsize)
    end
    dudt_ = let X=X, re=n.re, in=n.in, hidden=n.hidden, preprocess=n.preprocess
        (u,p,t) -> dudt_CDE(u,p,t,X,re,in,hidden, preprocess)
    end
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    solve(prob,n.args...;sense=n.sense, n.kwargs...)
end
function (n::NeuralCDE)(X::LinearInterpolation, p=n.p, u₀=get_u₀(X,n))
    x = nograd(X, f=n.preprocess)
    tstops = eltype(n.u₀).(collect(X.t))|> sort
    dudt_ = let X=X, re=n.re, in=n.in, hidden=n.hidden, preprocess=n.preprocess
        (u,p,t) -> dudt_CDE(u,p,t,X,re,in,hidden, preprocess)
    end
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    solve(prob,n.args...;sense=n.sense, tstops=tstops, n.kwargs...)
end

function (n::NeuralCDE{M})(X::T, p=n.p, u₀=get_u₀(X,n)) where {T<:Union{CubicSpline,CubicSplineRegularGrid},M<:Union{FastLayer,FastChain}}

    dudt_ = let X=X, model=n.model, in=n.in, hidden=n.hidden, preprocess=n.preprocess
        (u,p,t) -> Fast_dudt_CDE(u,p,t,X,model,in,hidden, preprocess)
    end

    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    solve(prob,n.args...;sense=n.sense, n.kwargs...)
end

function (n::NeuralCDE{M})(X::LinearInterpolation, p=n.p, u₀=get_u₀(X,n)) where {M<:Union{FastLayer,FastChain}}
    
    dudt_ = let X=X, model=n.model, in=n.in, hidden=n.hidden, preprocess=n.preprocess
        (u,p,t) -> Fast_dudt_CDE(u,p,t,X,model,in,hidden, preprocess)
    end

    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    solve(prob,n.args...;sense=n.sense, tstops=tstops, n.kwargs...)
end