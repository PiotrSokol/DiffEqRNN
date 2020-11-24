import DataInterpolations: munge_data, AbstractInterpolation, LinearInterpolation, CubicSpline, ConstantInterpolation
import LinearAlgebra:Tridiagonal
import Zygote:ignore

# Cubic Spline Interpolation
struct CubicSplineFixedGrid{uType,tType,RangeType,zType,FT,T} <: AbstractInterpolation{FT,T}
  u::uType
  t₀::tType
  t₁::tType
  Δt::tType
  t::RangeType
  z::zType
  CubicSplineFixedGrid{FT}(u,t₀,t₁,Δt,z) where FT = new{typeof(u),eltype(u),typeof(t₀:Δt:t₁),typeof(z),FT,eltype(u)}(u,t₀,t₁,Δt,t₀:Δt:t₁,z)
end

function CubicSplineFixedGrid(u::AV,t₀::T=0,t₁::T=length(u)-1,Δt::T=1) where {T<:Number,AV<:AbstractVector{<:Number}}
  @assert ~any(ismissing, u)
  n = length(u) - 1
  @assert length(t₀:Δt:t₁-Δt) == n
  h = vcat( zero(eltype(u)), ones(eltype(u),n), zero(eltype(u)))
  dl = h[2:n+1]
  d_tmp = 2 .* (h[1:n+1] .+ h[2:n+2])
  du = h[2:n+1]
  tA = Tridiagonal(dl,d_tmp,du)
  d = map(i -> i == 1 || i == n + 1 ? 0 : 6(u[i+1] - 2u[i] + u[i-1]), 1:n+1)
  z = tA\d
  CubicSplineFixedGrid{true}(u,t₀,t₁,Δt,z)
end

function CubicSplineFixedGrid(U::AV,t₀::T=0,t₁::T=size(U,2)-1,Δt::T=1) where {T<:Number,AV<:AbstractMatrix{<:Number}}
  @assert ~any(ismissing, U)
  n = size(U,2) - 1
  @assert length(t₀:Δt:t₁-Δt) == n
  h = vcat( zeros(eltype(U)), ones(eltype(U),n), zeros(eltype(U)))
  du = dl = copy(h[2:n+1])
  d_tmp = 2 .* (h[1:n+1] .+ h[2:n+2])

  tA = Tridiagonal(dl,d_tmp,du)
  d = reduce(hcat, map(i -> i == 1 || i == n + 1 ? zeros(eltype(U), size(U,1)) : 6(view(U, :, i+1) - 2view(U, :, i) + view(U, :, i-1)  ), 1:n+1) )
  z = tA\d'
  CubicSplineFixedGrid{true}(U,t₀,t₁,Δt,z)
end

function (A::CubicSplineFixedGrid{<:AbstractVector{<:Number}})(t::Number)
  re = t%A.Δt
  re /=A.Δt
  i = floor(Int64,t/A.Δt)
  i == i >= length(A.t) ? i = length(A.t) - 1 : nothing
  i == 0 ? i += 1 : nothing
  z(i) = A.z[i]
  u(i) = A.u[i]
  I = z(i) .* (A.Δt - re)^3 /6 .+ z(i+1) .* (re)^3 /6
  C = (u(i+1) .- z(i+1)./6).*(re)
  D = (u(i) .- z(i)./6).*(A.Δt - re)
  I + C + D
end

function (A::CubicSplineFixedGrid{<:AbstractMatrix{<:Number}})(t::Number)
  re = t%A.Δt
  re /=A.Δt
  i = floor(Int64,t/A.Δt)
  i == i >= length(A.t) ? i = length(A.t) - 1 : nothing
  i == 0 ? i += 1 : nothing
  u(i) = view(A.u, :,i)
  z(i) = view(A.z, i,:)
  I = z(i) .* (A.Δt - re)^3 /6 .+ z(i+1) .* (re)^3 /6
  C = (u(i+1) .- z(i+1)./6).*(re)
  D = (u(i) .- z(i)./6).*(A.Δt - re)
  I + C + D
end


function LinearInterpolationFixedGrid(u::AV,t₀::T=0,t₁::T=length(u),Δt::T=1) where {T<:Number,AV<:AbstractVector{<:Number}}
    t₀,t₁,Δt = promote(t₀,t₁,Δt)
    n = length(u)
    @assert length(t₀:Δt:t₁-Δt) == n
    t = collect(t₀:Δt:t₁-Δt)
    u, t = munge_data(u, t)
    return LinearInterpolation{true}(u,t)
end

function LinearInterpolationFixedGrid(U::AV,t₀::T=0,t₁::T=size(U,2),Δt::T=1) where {T<:Number,AV<:AbstractMatrix{<:Number}}
    t₀,t₁,Δt = promote(t₀,t₁,Δt)
    n = size(U,2) 
    @assert length(t₀:Δt:t₁-Δt) == n
    t = collect(t₀:Δt:t₁-Δt)
    u, t = munge_data(U, t)
    return LinearInterpolation{true}(U,t)
end

function ConstantInterpolationFixedGrid(u::AV,t₀::T=0,t₁::T=length(u),Δt::T=1) where {T<:Number,AV<:AbstractVector{<:Number}}
    t₀,t₁,Δt = promote(t₀,t₁,Δt)
    n = length(u)
    @assert length(t₀:Δt:t₁-Δt) == n
    t = collect(t₀:Δt:t₁-Δt)
    u, t = munge_data(u, t)
    return ConstantInterpolation{true}(u,t,:left)
end

function ConstantInterpolationFixedGrid(U::AV,t₀::T=0,t₁::T=size(U,2),Δt::T=1) where {T<:Number,AV<:AbstractMatrix{<:Number}}
    t₀,t₁,Δt = promote(t₀,t₁,Δt)
    n = size(U,2)
    @assert length(t₀:Δt:t₁-Δt) == n
    t = collect(t₀:Δt:t₁-Δt)
    u, t = munge_data(U, t)
    return ConstantInterpolation{true}(U,t,:left)
end

"""
Instructs Zygote to ignore the following code block.
Analogous to with torch.nograd(): context in Python
"""


const UnivInpt = Union{LinearInterpolation{T},ConstantInterpolation{T},CubicSpline{T},CubicSplineFixedGrid{T}} where  T<:AbstractVector{<:Number}
const MultivInpt = Union{LinearInterpolation{T},ConstantInterpolation{T},CubicSpline{T},CubicSplineFixedGrid{T}} where  T<:AbstractMatrix{<:Number}

struct nograd{T}
    interpolant::T
    dtype
    t
    f
    function nograd(interp::ITP; f = identity ) where {ITP<:UnivInpt}
        new{typeof(interp)}(interp,eltype(interp.u),collect(interp.t),f)
    end
    
    function nograd(interp::ITP; f = permutedims ) where {ITP<:MultivInpt}
        new{typeof(interp)}(interp,eltype(interp.u),collect(interp.t),f)
    end
end


function (n::nograd{<:UnivInpt})(t)
 x = ignore() do
    n.interpolant(t) |> n.f
  end
end

function (n::nograd{<:MultivInpt})(t)
 x = ignore() do
    n.interpolant(t) |> n.f
  end
end

function get_batchsize(n::nograd)
  size(n.interpolant.u)[1]
end
"""
Fixes extrapolation issue -- DataInterpolations.jl prevents linear interpolation.
"""
function (A::LinearInterpolation{<:AbstractVector{<:Number}})(t::Number)
  idx = findfirst(x->x>=t,A.t)
  idx == nothing ? idx = length(A.t) - 1 : idx -= 1
  idx == 0 ? idx += 1 : nothing
  θ = (t - A.t[idx])/ (A.t[idx+1] - A.t[idx])
  (1-θ)*A.u[idx] + θ*A.u[idx+1]
end

function (A::LinearInterpolation{<:AbstractMatrix{<:Number}})(t::Number)
  idx = findfirst(x->x>=t,A.t)
  idx == nothing ? idx = length(A.t) - 1 : idx -= 1
  idx == 0 ? idx += 1 : nothing
  θ = (t - A.t[idx])/ (A.t[idx+1] - A.t[idx])
  (1-θ)*A.u[:,idx] + θ*A.u[:,idx+1]
end
