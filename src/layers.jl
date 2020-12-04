include("utils.jl")
abstract type AbstractRNNDELayer <: Function end
using Flux
using Base
import Flux: functor, gate
import Flux: kaiming_normal, glorot_uniform
const ℕ = kaiming_normal
"""
Vanilla RNN
"""
struct ∂RNNCell{F,A,V,S} <:AbstractRNNDELayer
  σ::F
  Wᵢ::A
  Wᵣ::A
  b::V
  u₀::S
end

function ∂RNNCell(in::Integer, out::Integer; σ = tanh,
      initWᵢ = ℕ, initWᵣ=limit_cycle, initb = Flux.zeros, inits=state0_init)
      Wᵢ = initWᵢ(out,in)
      Wᵣ = initWᵣ(out,out)
      b = initb(out)
      u₀ = state0_init(out,1)
      ∂RNNCell(σ,Wᵢ,Wᵣ,b,u₀)
end
function ∂RNNCell(in::Integer, out::Integer, init)
      Wᵢ = init(out,in)
      Wᵣ = init(out,out)
      b = init(out)
      u₀ = Flux.zeros(out,1)
      ∂RNNCell(σ,Wᵢ,Wᵣ,b,u₀)
end

function (m::∂RNNCell)(h, x)
  σ, Wᵢ, Wᵣ, b = m.σ, m.Wᵢ, m.Wᵣ, m.b
  ḣ = σ.(Wᵢ*x .+ Wᵣ*h .+ b).-h
  return ḣ
end

Flux.@functor ∂RNNCell
Flux.trainable(m::∂RNNCell) = (m.Wᵢ, m.Wᵣ, m.b,)

function Base.show(io::IO, l::∂RNNCell)
  print(io, "∂RNNCell(", size(l.Wᵢ, 2), ", ", size(l.Wᵢ, 1))
  l.σ == identity || print(io, ", ", l.σ)
  print(io, ")")
end


"""
GRU
"""
struct ∂GRUCell{A,V,S} <:AbstractRNNDELayer
  Wᵢ::A
  Wᵣ::A
  b::V
  u₀::S
end
function ∂GRUCell(in::Integer, out::Integer; initWᵢ = ℕ, initWᵣ = (dims...)  -> limit_cycle(dims... ,σ=3.0f0), initWᵣᵤ = Flux.zeros, initb = Flux.zeros, inits= init=state0_init)
  Wᵢ=initWᵢ(out * 3, in)
  Wᵣ = vcat( initWᵣᵤ(2*out,out), initWᵣ(out,out) )
  b = initb(out * 3)
  u₀ = state0_init(out,1)
  ∂GRUCell(Wᵢ,Wᵣ,b,u₀)
end
function ∂GRUCell(in::Integer, out::Integer, init)
  Wᵢ=init(out * 3, in)
  Wᵣ = vcat( init(2*out,out), init(out,out) )
  b = init(out * 3)
  u₀ = Flux.zeros(out,1)
  ∂GRUCell(Wᵢ,Wᵣ,b,u₀)
end
function (m::∂GRUCell)(h, x)
  b, o = m.b, size(h, 1)
  gx, gh = m.Wᵢ*x, m.Wᵣ*h
  r = σ.(gate(gx, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1))
  z = σ.(gate(gx, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2))
  h̃ = tanh.(gate(gx, o, 3) .+ r.* gate(gh, o, 3) .+ gate(b, o, 3))
  ḣ =  (z .- 1).* (h .- h̃)
  return ḣ
end

Flux.@functor ∂GRUCell
Flux.trainable(m::∂GRUCell) = (m.Wᵢ, m.Wᵣ, m.b,)

Base.show(io::IO, l::∂GRUCell) =
  print(io, "∂GRUCell(", size(l.Wᵢ, 2), ", ", size(l.Wᵢ, 1)÷3, ")")

"""
Continuous time LSTM
"""
struct ∂LSTMCell{A,V,S} <:AbstractRNNDELayer
  Wᵢ::A
  Wᵣ::A
  b::V
  u₀::S
end

function ∂LSTMCell(in::Integer, out::Integer;
                  init = glorot_uniform,
                  initb = zeros, inits = Flux.zeros)
  cell = ∂LSTMCell(init(out * 4, in), init(out * 4, out), initb(out * 4),
                    inits(2out,1))
  cell.b[gate(out, 2)] .= 2
  return cell
end

function (m::∂LSTMCell)(hc, x)
  b, o = m.b, size(hc, 1)÷2
  h, c = gate(hc, o, 1), gate(hc, o, 2)
  g = m.Wᵢ*x .+ m.Wᵣ*h .+ b
  input = σ.(gate(g, o, 1))
  forget = σ.(gate(g, o, 2))
  cell = tanh.(gate(g, o, 3))
  output = σ.(gate(g, o, 4))
  ċ = forget .* c .+ input .* cell .- c
  ḣ = output .* tanh.(c) .- h
  return vcat(ḣ, ċ)
end

Flux.@functor ∂LSTMCell
Flux.trainable(m::∂LSTMCell) = (m.Wᵢ, m.Wᵣ, m.b,)

Base.show(io::IO, l::∂LSTMCell) =
  print(io, "∂LSTMCell(", size(l.Wᵢ, 2), ", ", size(l.Wᵢ, 1)÷4, ")")