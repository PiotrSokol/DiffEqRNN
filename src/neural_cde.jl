
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
    function NeuralCDE(model,tspan, nin::Int, nhidden::Int, args...;p = nothing, preprocess=nothing, sense = InterpolatingAdjoint(autojacvec=ZygoteVJP()), kwargs...)
        _p,re = destructure(model)
        if isnothing(p)
            p = _p
        end
        if isnothing(preprocess)
            preprocess = x-> reshape(x,1, nin, :)
        end

        new{typeof(model),typeof(p),typeof(re),
            typeof(tspan),typeof(args),typeof(kwargs),
            typeof(hidden)}(
            model,p,re,tspan,args,kwargs,sense,nin,
            nhidden, preprocess)
    end
end
function (n::NeuralCDE)(X::T; u₀=nothing, p=n.p) where {T<:Union{CubicSpline,CubicSplineRegularGrid}}
    x = nograd(X, f=n.preprocess)
    if isnothing(u₀)
        u₀ = repeat(n.u₀, 1, get_batchsize(x)) |> deepcopy
    end
    function du_dt_(u,p,t)
        u = reshape( n.re(p)(u), n.in, n.hidden,:)
        dX = derivative(x,t)
        du = batched_mul(dX,u)[1,:,:]
    end
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    solve(prob,n.args...;sense=n.sense, n.kwargs...)
end
function (n::NeuralCDE)(X::LinearInterpolation; u₀=nothing, p=n.p)
    x = nograd(X, f=n.preprocess)
    tstops = eltype(n.u₀).(collect(X.t))|> sort
    if isnothing(u₀)
        u₀ = repeat(n.u₀, 1, get_batchsize(x)) |> deepcopy
    end
    function du_dt_(u,p,t)
        u = reshape( n.re(p)(u), n.in, n.hidden,:)
        dX = derivative(x,t)
        du = batched_mul(dX,u)[1,:,:]
    end
    ff = ODEFunction{false}(dudt_,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,u₀,getfield(n,:tspan),p)
    solve(prob,n.args...;sense=n.sense, tstops=tstops, n.kwargs...)
end