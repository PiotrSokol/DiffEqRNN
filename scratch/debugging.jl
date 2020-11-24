using DiffEqFlux, OrdinaryDiffEq, Flux, Optim
include("/Users/piotrsokol/Documents/RNNODE.jl/src/rnn_ode.jl")
ipt = sqrt(1/2)randn(Float32,3,100)
lpt = LinearInterpolationFixedGrid(ipt);
cpt = ConstantInterpolationFixedGrid(ipt);
spt = CubicSplineFixedGrid(ipt);
∂nn = ∂GRUCell(1,2)
tspan = [0.f0,100.f0]
tsteps = collect(tspan[1] : tspan[2])
node = RNNODE(∂nn, [0.f0,100.f0], saveat=tsteps);
sol = node(node.u₀);