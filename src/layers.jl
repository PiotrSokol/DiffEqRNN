include("../utils.jl")
abstract type AbstractRNNDELayer <: Function end
import Flux: functor, gate
"""
Vanilla RNN
"""
mutable struct ∂RNNCell{F,A,V} <:AbstractRNNDELayer
  σ::F
  Wᵢ::A
  Wᵣ::A
  b::V
end

∂RNNCell(in::Integer, out::Integer, σ = tanh;
        init = he_normal, initWᵣ=limit_cycle ) =
  ∂RNNCell(σ, init(out, in), initWᵣ(out, out),
          zeros(out))

function (m::∂RNNCell)(h, x)
  σ, Wᵢ, Wᵣ, b = m.σ, m.Wᵢ, m.Wᵣ, m.b
  ḣ = σ.(Wi*x .+ Wh*h .+ b).-h
  return ḣ
end

@Flux.functor ∂RNNCell

"""
GRU
"""
struct ∂GRUCell{A,V} <:AbstractRNNDELayer
  Wᵢ::A
  Wᵣ::A
  b::V
end

function ∂GRUCell(in, out; initWᵢ = he_normal, initWᵣ = limit_cycle, initWᵣᵤ = zeros, initb = zeros)
  ∂GRUCell(
  initWᵢ(out * 3, in),
  vcat( initWᵣᵤ(2*out,in), initWᵣ(out,in) ),
  initb(out * 3)
  )
end

∂GRUCell(in, out, init) = ∂GRUCell(in, out; initWᵢ = init, initWᵣ = init, initWᵣᵤ = init, initb = init)

function (m::∂GRUCell)(h, x)
  b, o = m.b, size(h, 1)
  gx, gh = m.Wi*x, m.Wh*h
  r = σ.(gate(gx, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1))
  z = σ.(gate(gx, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2))
  h̃ = tanh.(gate(gx, o, 3) .+ r.* gate(gh, o, 3) .+ gate(b, o, 3))
  ḣ =  (z .- 1).* (h .- h̃)
  return ḣ
end

@Flux.functor ∂GRUCell
