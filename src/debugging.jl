using Plots
include("/Users/piotrsokol/Documents/RNNODE.jl/src/rnn_ode.jl")
∂nn = ∂GRUCell(1,2)
node = RNNODE(∂nn, [0.f0,100.f0])
u₀ = node.u₀
sol = node(rand(eltype(u₀), 2,1));

ipt = randn(2,100)
Lipt = LinearInterpolationFixedGrid(ipt);
Constipt = ConstantInterpolationFixedGrid(ipt);
Cubipt = CubicSplineFixedGrid(ipt);

node(Lipt)