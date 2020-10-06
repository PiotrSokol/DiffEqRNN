import DataInterpolations: munge_data, AbstractInterpolation, LinearInterpolation, CubicSpline
import LinearAlgebra:Tridiagonal

# Cubic Spline Interpolation
struct CubicSplineFixedGrid{uType,tType,RangeType,zType,FT,T} <: AbstractInterpolation{FT,T}
  u::uType
  t₀::tType
  t₁::tType
  Δt::tType
  t::RangeType
  z::zType
  CubicSplineFixedGrid{FT}(u,t₀,t₁,Δt,z) where FT = new{typeof(u),typeof(t₀),typeof(t₀:Δt:t₁),typeof(z),FT,eltype(u)}(u,t₀,t₁,Δt,t₀:Δt:t₁,z)
end

function CubicSplineFixedGrid(u::AV,t₀::T=0,t₁::T=length(u)-1,Δt::T=1) where {T<:Number,AV<:AbstractVector{<:Number}}
  @assert ~any(ismissing, u)
  t₀,t₁,Δt = promote(t₀,t₁,Δt)
  n = length(u) - 1
  @assert length(t₀:Δt:t₁-Δt) == n
  h = vcat( zero(eltype(u)), ones(eltype(u),n), zero(eltype(u)))
  dl = h[2:n+1]
  d_tmp = 2 .* (h[1:n+1] .+ h[2:n+2])
  du = h[2:n+1]
  tA = LinearAlgebra.Tridiagonal(dl,d_tmp,du)
  d = map(i -> i == 1 || i == n + 1 ? 0 : 6(u[i+1] - 2u[i] + u[i-1]), 1:n+1)
  z = tA\d
  CubicSplineFixedGrid{true}(u,t₀,t₁,Δt,z)
end

function CubicSplineFixedGrid(U::AV,t₀::T=0,t₁::T=size(U,2)-1,Δt::T=1) where {T<:Number,AV<:AbstractMatrix{<:Number}}
  # u, t = munge_data(u, t)
  t₀,t₁,Δt = promote(t₀,t₁,Δt)
  @assert ~any(ismissing, U)
  n = size(U,2) - 1
  @assert length(t₀:Δt:t₁-Δt) == n
  h = vcat( zeros(eltype(U)), ones(eltype(U),n), zeros(eltype(U)))
  du = dl = copy(h[2:n+1])
  d_tmp = 2 .* (h[1:n+1] .+ h[2:n+2])

  tA = LinearAlgebra.Tridiagonal(dl,d_tmp,du)
  d = reduce(hcat, map(i -> i == 1 || i == n + 1 ? zeros(eltype(U), size(U,1)) : 6(view(U, :, i+1) - 2view(U, :, i) + view(U, :, i-1)  ), 1:n+1) )
  z = tA\d'
  CubicSplineFixedGrid{true}(U,t₀,t₁,Δt,z)
end

function (A::CubicSplineFixedGrid{<:AbstractVector{<:Number}})(t::Number)
  re = t%A.Δt
  re /=A.Δt
  i = floor(Int64,t/A.Δt)
  i == i > length(A.t) ? i = length(A.t) - 1 : nothing
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
  i == i > length(A.t) ? i = length(A.t) - 1 : nothing
  i == 0 ? i += 1 : nothing
  u(i) = view(B.u, :,i)
  z(i) = view(B.z, i,:)
  I = z(i) .* (A.Δt - re)^3 /6 .+ z(i+1) .* (re)^3 /6
  C = (u(i+1) .- z(i+1)./6).*(re)
  D = (u(i) .- z(i)./6).*(A.Δt - re)
  I + C + D
end


function derivative(A::LinearInterpolation{<:AbstractVector{<:Number}}, t::Number)
    idx = findfirst(x -> x >= t, A.t) - 1
    idx == 0 ? idx += 1 : nothing
    θ = 1 / (A.t[idx+1] - A.t[idx])
    (A.u[idx+1] - A.u[idx]) / (A.t[idx+1] - A.t[idx])
end

function derivative(A::LinearInterpolation{<:AbstractMatrix{<:Number}}, t::Number)
    idx = findfirst(x -> x >= t, A.t) - 1
    idx == 0 ? idx += 1 : nothing
    θ = 1 / (A.t[idx+1] - A.t[idx])
    (A.u[:, idx+1] - A.u[:, idx]) / (A.t[idx+1] - A.t[idx])
end

function derivative(A::CubicSpline{<:AbstractVector{<:Number}}, t::Number)
    i = findfirst(x -> x >= t, A.t)
    i == nothing ? i = length(A.t) - 1 : i -= 1
    i == 0 ? i += 1 : nothing
    dI = -3A.z[i] * (A.t[i + 1] - t)^2 / (6A.h[i + 1]) + 3A.z[i + 1] * (t - A.t[i])^2 / (6A.h[i + 1])
    dC = A.u[i + 1] / A.h[i + 1] - A.z[i + 1] * A.h[i + 1] / 6
    dD = -(A.u[i] / A.h[i + 1] - A.z[i] * A.h[i + 1] / 6)
    dI + dC + dD
end

function derivative(C::CubicSplineFixedGrid{<:AbstractVector{<:Number}}, t::Number)
    re = t%C.Δt
    re /=C.Δt
    i = Int(floor(t/C.Δt))
    i == i > length(C.t) ? i = length(C.t) - 1 : nothing
    i == 0 ? i += 1 : nothing
    dI = -3C.z[i] * (C.Δt - re)^2 / 6 + 3C.z[i + 1] * (re)^2 / 6
    dC = C.u[i + 1] - C.z[i + 1] / 6
    dD = -(C.u[i]- C.z[i]/ 6)
    dI + dC + dD
end

function derivative(B::CubicSplineFixedGrid{<:AbstractMatrix{<:Number}}, t::Number)
    re = t%B.Δt
    re /=B.Δt
    i = Int(floor(t/B.Δt))
    i == i > length(B.t) ? i = length(B.t) - 1 : nothing
    i == 0 ? i += 1 : nothing
    u(i) = view(B.u, :,i)
    z(i) = view(B.z, i,:)
    dI = -3z(i) .* (B.Δt - re)^2 /6 .+ 3z(i+1) .* (re)^2 / 6
    dC = u(i+1) .- z(i+1)./ 6
    dD = -(u(i) .- z(i)./ 6)
    dI .+ dC .+ dD
end
