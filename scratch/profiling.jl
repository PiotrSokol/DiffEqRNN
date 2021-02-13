using Profile, BenchmarkTools
using DiffEqRNN, Random, Flux, OrdinaryDiffEq, DiffEqFlux

Random.seed!(0)
t₁ = 75
bs = 100
x = Float32(sqrt(1/2))randn(Float32, bs, t₁)
cells = [∂RNNCell, ∂GRUCell, ∂LSTMCell]
interpolators = [CubicSplineRegularGrid, LinearInterpolationRegularGrid, ConstantInterpolationRegularGrid]
##
itp = interpolators[1]
cell = cells[1]
##
X = itp(x)
∂nn = cell(1,250)
tspan = Float32.([0, t₁])
tsteps = collect(tspan[1] : tspan[2])
node = RNNODE(∂nn, tspan, Tsit5(), reltol=1e-4,abstol=1e-4, saveat=tsteps, preprocess=permutedims )
u0 = repeat(node.u₀, 1, 100)
predict_neuralode(p) = Array(node(X,p))
loss_neuralode(p) = sum(abs2, predict_neuralode(p))
loss_before = loss_neuralode(node.p)
optim = ADAM(0.05)
DiffEqFlux.sciml_train(loss_neuralode, node.p, optim, maxiters = 1)
##
@benchmark DiffEqFlux.sciml_train(loss_neuralode, node.p, optim, maxiters = 5)
##
Profile.clear()
@profile DiffEqFlux.sciml_train(loss_neuralode, node.p, optim, maxiters = 5)
# @profview
Juno.profiler()
##
Profile.clear()
du_dt = Dense(250,250,σ)
ode = NeuralODE(du_dt,tspan, Tsit5(), reltol=1e-4,abstol=1e-4, saveat=tsteps) ## took about 60 seconds
u0 = rand(Float32,250,bs)
predict_ode(p) = Array(ode(u0, p))
loss_ode(p) = sum(abs2, predict_ode(p))
##
loss_before = loss_ode(ode.p)
DiffEqFlux.sciml_train(loss_ode, ode.p, optim, maxiters = 1)
##
@benchmark DiffEqFlux.sciml_train(loss_ode, ode.p, optim, maxiters = 5)  # took 15 seconds
Profile.clear()
@profile DiffEqFlux.sciml_train(loss_ode, ode.p, optim, maxiters = 5)
##
predict_node2(p) = Array(node(u0,p))
loss_node2(p) = sum(abs2, predict_node2(p))
optim = ADAM(0.05)
DiffEqFlux.sciml_train(loss_node2, node.p, optim, maxiters = 1)
@benchmark DiffEqFlux.sciml_train(loss_node2, node.p, optim, maxiters = 5)
@profile DiffEqFlux.sciml_train(loss_node2, node.p, optim, maxiters = 5)
Juno.profiler()
