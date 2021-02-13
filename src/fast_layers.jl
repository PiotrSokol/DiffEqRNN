using DiffEqRNN
using DiffEqRNN:limit_cycle,state0_init
using Flux:kaiming_normal, gate, glorot_uniform
using Flux
using DiffEqFlux:FastLayer, FastChain, FastDense, initial_params
import DiffEqFlux:initial_params,paramlength
abstract type FastRNNLayer <: FastLayer end
initial_params(f::T) where {T<:FastRNNLayer} = f.initial_params()

##
struct Fast∂RNNCell{F,F2,V} <: FastRNNLayer
  out::Int
  in::Int
  σ::F
  initial_params::F2
  u₀::V
    function Fast∂RNNCell(in::Integer, out::Integer, σ = tanh;
                    initWᵢ = kaiming_normal, initWᵣ=limit_cycle, initb = Flux.zeros)
        Wᵢ = initWᵢ(out,in)
        Wᵣ = initWᵣ(out,out)
        b = initb(out)
        u₀ = state0_init(out)
        temp = mapreduce(vec, vcat, (Wᵢ,Wᵣ,b))
        initial_params() = temp
        new{typeof(σ),typeof(initial_params),typeof(u₀)}(out,in,σ,initial_params,u₀)
    end
    function Fast∂RNNCell(in::Integer, out::Integer, σ, init)
        Wᵢ = init(out,in)
        Wᵣ = init(out,out)
        b = init(out)
        u₀ = state0_init(out)

        temp = mapreduce(vec, vcat, (Wᵢ,Wᵣ,b))
        initial_params() = temp
        new{typeof(σ),typeof(initial_params),typeof(u₀)}(out,in,σ,initial_params,u₀)
    end
end

function (f::Fast∂RNNCell)(h,x,p)
    Wᵢ = @views reshape(p[1:f.out*f.in],f.out, f.in)
    Wᵣ = @views reshape(p[f.out*f.in+1:f.out*(f.in+f.out)],f.out,f.out)
    b = @views p[end-f.out+1:end]
    ḣ = f.σ.((Wᵢ*x .+ Wᵣ*h .+ b)).-h
end
function (f::Fast∂RNNCell)(h,p)
    # Wᵢ = @views p[1:f.out*f.in]
    Wᵣ = @views reshape(p[f.out*f.in+1:f.out*(f.in+f.out)],f.out,f.out)
    b = @views p[end-f.out+1:end]
    ḣ = f.σ.(( Wᵣ*h .+ b)).-h
end
paramlength(f::Fast∂RNNCell) = f.out*(f.out+f.in+1)
##
struct Fast∂GRUCell{F2,V} <: FastRNNLayer
  out::Int
  in::Int
  initial_params::F2
  u₀::V
    function Fast∂GRUCell(in::Integer, out::Integer; initWᵢ = kaiming_normal, initWᵣ = (dims...)  -> limit_cycle(dims... ,σ=3.0f0), initWᵣᵤ = zeros, initb = zeros)
        Wᵢ = initWᵢ(out * 3, in)
        Wᵣ = vcat( initWᵣᵤ(2*out,out), initWᵣ(out,out) )
        b = initb(out * 3)
        u₀ = state0_init(out)

        temp = mapreduce(vec, vcat, (Wᵢ,Wᵣ,b))
        initial_params() = temp
        new{typeof(initial_params),typeof(u₀)}(out,in,initial_params,u₀)
    end
    function Fast∂GRUCell(in::Integer, out::Integer, init)
        Wᵢ = init(out * 3, in)
        Wᵣ = vcat( init(2*out,out), init(out,out) )
        b = init(out * 3)
        u₀ = state0_init(out)

        temp = mapreduce(vec, vcat, (Wᵢ,Wᵣ,b))
        initial_params() = temp
        new{typeof(initial_params),typeof(u₀)}(out,in,initial_params,u₀)
    end
end
function (f::Fast∂GRUCell)(h,x,p)
    Wᵢ = reshape(p[1:3f.out*f.in], 3f.out, f.in)
    Wᵣ = reshape(p[1+(3f.out*f.in):3f.out*(f.in+f.out)], 3f.out, f.out)
    b = @views p[end-3f.out+1:end]
    gx, gh = Wᵢ*x, Wᵣ*h
    r = σ.(gate(gx, f.out, 1) .+ gate(gh, f.out, 1) .+ gate(b, f.out, 1))
    z = σ.(gate(gx, f.out, 2) .+ gate(gh, f.out, 2) .+ gate(b, f.out, 2))
    h̃ = tanh.(gate(gx, f.out, 3) .+ r.* gate(gh, f.out, 3) .+ gate(b, f.out, 3))
    ḣ =  (z .- 1).* (h .- h̃)
    return ḣ
end
function (f::Fast∂GRUCell)(h,p)
    Wᵣ = reshape(p[1+(3f.out*f.in):3f.out*(f.in+f.out)], 3f.out, f.out)
    b = @views p[end-3f.out+1:end]
    gh = Wᵣ*h
    r = σ.(gate(gh, f.out, 1) .+ gate(b, f.out, 1))
    z = σ.(gate(gh, f.out, 2) .+ gate(b, f.out, 2))
    h̃ = tanh.(r.* gate(gh, f.out, 3) .+ gate(b, f.out, 3))
    ḣ =  (z .- 1).* (h .- h̃)
    return ḣ
end
paramlength(f::Fast∂GRUCell) = 3f.out*(f.out+f.in+1)
##
struct Fast∂LSTMCell{F2,V} <: FastRNNLayer
  out::Int
  in::Int
  initial_params::F2
  u₀::V
    function Fast∂LSTMCell(in::Integer, out::Integer;
                  init = glorot_uniform,
                  initb = zeros)
        Wᵢ = init(out * 4, in)
        Wᵣ = init(out * 4, out)
        b = initb(out * 4)
        u₀ = state0_init(out)
        b[gate(out, 2)] .=2
        u₀ = state0_init(2out)
        
        temp = mapreduce(vec, vcat, (Wᵢ,Wᵣ,b))
        initial_params() = temp
        new{typeof(initial_params),typeof(u₀)}(out,in,initial_params,u₀)
    end
end
function (f::Fast∂LSTMCell)(hc,x,p)
    Wᵢ = reshape(p[1:4f.out*f.in], 4f.out, f.in)
    Wᵣ = reshape(p[1+(4f.out*f.in):4f.out*(f.in+f.out)], 4f.out, f.out)
    b = @views p[end-4f.out+1:end]

    h, c = gate(hc, f.out, 1), gate(hc, f.out, 2)
    g = Wᵢ*x .+ Wᵣ*h .+ b
    input = σ.(gate(g, f.out, 1))
    forget = σ.(gate(g, f.out, 2))
    cell = tanh.(gate(g, f.out, 3))
    output = σ.(gate(g, f.out, 4))
    ċ = forget .* c .+ input .* cell .- c
    ḣ = output .* tanh.(c) .- h
    return vcat(ḣ, ċ)
end
function (f::Fast∂LSTMCell)(hc,p)
    Wᵢ = reshape(p[1:4f.out*f.in], 4f.out, f.in)
    Wᵣ = reshape(p[1+(4f.out*f.in):4f.out*(f.in+f.out)], 4f.out, f.out)
    b = @views p[end-4f.out+1:end]

  h, c = gate(hc, f.out, 1), gate(hc, f.out, 2)
  g = Wᵣ*h .+ b
  input = σ.(gate(g, f.out, 1))
  forget = σ.(gate(g, f.out, 2))
  cell = tanh.(gate(g, f.out, 3))
  output = σ.(gate(g, f.out, 4))
  ċ = forget .* c .+ input .* cell .- c
  ḣ = output .* tanh.(c) .- h
  return vcat(ḣ, ċ)
end
paramlength(f::Fast∂LSTMCell) = 4f.out*(f.out+f.in+1)

# # ##
# nhidden = 14
# nbatch = 17
# nin = 13
# rnn = ∂RNNCell(nin,nhidden)
# p = mapreduce(vec, vcat, params(rnn))
# fastrnn = Fast∂RNNCell(nin,nhidden)
# ##
# p0 = initial_params(fastrnn)
# ##
# if typeof(rnn) <: ∂LSTMCell
#     nhidden*=2
# end
# u0 = randn(Float32,nhidden,nbatch)
# x = randn(Float32, nin,nbatch)

# h1 = rnn(u0,x)
# fasth = fastrnn(u0,x,p)
# h1 ≈ fasth

##
