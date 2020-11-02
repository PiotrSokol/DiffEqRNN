using Statistics
using DataInterpolations
using Flux
using OrdinaryDiffEq
using DiffEqSensitivity
using Zygote
using ProgressBars
using Random

T = Float32

bs = 512
X = [rand(T, 10, 50) for _ in 1:bs*10]

function create_spline(i)
    x = X[i]
    t = x[end, :]
    t = (t .- minimum(t)) ./ (maximum(t) - minimum(t))

    spline = QuadraticInterpolation(x, t)
end

splines = [create_spline(i) for i in tqdm(1:length(X))]

rand_inds = randperm(length(X))

i_sz = size(X[1], 1)
h_sz = 16

use_gpu = false
batches = [[splines[rand_inds[(i-1)*bs+1:i*bs]]] for i in tqdm(1:length(X)÷bs)]

data_ = Iterators.cycle(batches)

function call_and_cat(splines, t)
    vals = Zygote.ignore() do
        vals = reduce(hcat,[spline(t) for spline in splines])
    end
    vals |> (use_gpu ? gpu : cpu)
end

function derivative(A::QuadraticInterpolation, t::Number)
    idx = findfirst(x -> x >= t, A.t) - 1
    idx == 0 ? idx += 1 : nothing
    if idx == length(A.t) - 1
        i₀ = idx - 1; i₁ = idx; i₂ = i₁ + 1;
    else
        i₀ = idx; i₁ = i₀ + 1; i₂ = i₁ + 1;
    end
    dl₀ = (2t - A.t[i₁] - A.t[i₂]) / ((A.t[i₀] - A.t[i₁]) * (A.t[i₀] - A.t[i₂]))
    dl₁ = (2t - A.t[i₀] - A.t[i₂]) / ((A.t[i₁] - A.t[i₀]) * (A.t[i₁] - A.t[i₂]))
    dl₂ = (2t - A.t[i₀] - A.t[i₁]) / ((A.t[i₂] - A.t[i₀]) * (A.t[i₂] - A.t[i₁]))
    @views @. A.u[:, i₀] * dl₀ + A.u[:, i₁] * dl₁ + A.u[:, i₂] * dl₂
end

function derivative_call_and_cat(splines, t)
    vals = Zygote.ignore() do
        reduce(hcat,[derivative(spline, t) for spline in splines]) |> (use_gpu ? gpu : cpu)
    end
end

cde = Chain(
    Dense(h_sz, h_sz, relu),
    Dense(h_sz, h_sz*i_sz, tanh),
) |> (use_gpu ? gpu : cpu)

h_to_out = Dense(h_sz, 2) |> (use_gpu ? gpu : cpu)

initial = Dense(i_sz, h_sz) |> (use_gpu ? gpu : cpu)

cde_p, cde_re = Flux.destructure(cde)
initial_p, initial_re = Flux.destructure(initial)
h_to_out_p, h_to_out_re = Flux.destructure(h_to_out)

basic_tgrad(u,p,t) = zero(u)

function predict_func(p, BX)
    By = call_and_cat(BX, 1)

    x0 = call_and_cat(BX, 0)
    i = 1
    j = (i-1)+length(initial_p)

    h0 = initial_re(p[i:j])(x0)

    function dhdt(h,p,t)
        x = derivative_call_and_cat(BX, t)
        bs = size(h, 2)
        a = reshape(cde_re(p)(h), (i_sz, h_sz, bs))
        b = reshape(x, (1, i_sz, bs))

        dh = batched_mul(b,a)[1,:,:]
    end

    i = j+1
    j = (i-1)+length(cde_p)

    tspan = (0.0f0, 0.8f0)

    ff = ODEFunction{false}(dhdt,tgrad=basic_tgrad)
    prob = ODEProblem{false}(ff,h0,tspan,p[i:j])
    sense = InterpolatingAdjoint(autojacvec=ZygoteVJP())
    solver = Tsit5()

    sol = solve(prob,solver,u0=h0,saveat=tspan[end], save_start=false, sensealg=sense)
    #@show sol.destats
    i = j+1
    j = (i-1)+length(h_to_out_p)

    y_hat = h_to_out_re(p[i:j])(sol[end])

    y_hat, By[1:2, :]
end

function loss_func(p, BX)
    y_hat, y = predict_func(p, BX)

    mean(sum(sqrt.((y .- y_hat).^2), dims=1))
end

p = vcat(initial_p, cde_p, h_to_out_p)

callback = function (p, l)
  display(l)
  return false
end

using DiffEqFlux

Zygote.gradient((p)->loss_func(p, first(data_)...),p)
@time Zygote.gradient((p)->loss_func(p, first(data_)...),p)

@time result_neuralode = DiffEqFlux.sciml_train(loss_func, p, ADAM(0.05),
    data_,
    cb = callback,
    maxiters = 10)
