
struct NeuralCDE{M,P,RE,T,A,K,I} <: NeuralDELayer
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
    u₀
    function NeuralCDE(model,tspan, nin::Int, nhidden::Int, args...;p = nothing, preprocess=nothing, u₀=nothing, sense = InterpolatingAdjoint(autojacvec=ZygoteVJP()), kwargs...)
        _p,re = destructure(model)
        if isnothing(p)
            p = _p
        end
        if isnothing(preprocess)
            preprocess = x-> reshape(x, nin, :)
        end
        if isnothing(u₀)
            u₀=Random.randn!(similar(p,nhidden))./eltype(p)(sqrt(nhidden))
        end

        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(args),typeof(kwargs),
            typeof(nhidden)}(
            model,p,re,tspan,args,kwargs,sense,nin,
            nhidden, preprocess, u₀)
    end
end
function dudt_CDE(u,p,t,X,re,in,hidden, preprocess)
        u = permutedims(reshape( re(p)(u), in, hidden,:), [2,1,3])
        dX = derivative(X,t) |> preprocess
        batched_vec(u,dX)
end
function (n::NeuralCDE)(X::T; u₀=nothing, p=n.p) where {T<:Union{CubicSpline,CubicSplineRegularGrid}}
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
function (n::NeuralCDE)(X::LinearInterpolation; u₀=nothing, p=n.p)
    x = nograd(X, f=n.preprocess)
    tstops = eltype(n.u₀).(collect(X.t))|> sort
    if isnothing(u₀)
        u₀ = repeat(n.u₀, 1, infer_batchsizes(x))
    end
    dudt_ = let X=X, re=n.re, in=n.in, hidden=n.hidden, preprocess=n.preprocess
        (u,p,t) -> dudt_CDE(u,p,t,X,re,in,hidden, preprocess)
    end
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    solve(prob,n.args...;sense=n.sense, tstops=tstops, n.kwargs...)
end