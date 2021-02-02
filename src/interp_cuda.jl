using .CUDA
function CubicSplineRegularGrid(U::CuMatrix{T2}; t₀::T=0, t₁::T=size(U,2)-1,Δt::T=1) where {T<:Number,T2<:Number}
  @assert ~any(ismissing, U)
  n = size(U,2) - 1
  @assert length(t₀:Δt:t₁-Δt) == n
  h = vcat( zeros(eltype(U)), Δt*ones(eltype(U),n), zeros(eltype(U)))
  du = dl = copy(h[2:n+1])
  d_tmp = 2 .* (h[1:n+1] .+ h[2:n+2])

  tA = convert(CuArray,Tridiagonal(dl,d_tmp,du))
  d = hcat(CUDA.zeros(eltype(U), size(U)[1]),  6diff(diff(U, dims=2), dims=2)./Δt, CUDA.zeros(eltype(U), size(U)[1]) )
  z = tA\permutedims(d) |> permutedims
  CubicSplineRegularGrid{true}(U,t₀,t₁,Δt,z)
end

function DataInterpolations._interpolate(A::CubicSplineRegularGrid{<:CuArray{<:Number}}, t::Number)
  re = eltype(A.u)(t%A.Δt)
  i = floor(Int,t/A.Δt + 1)
  i == i >= length(A.t) ? i = length(A.t) - 1 : nothing
  i == min(1,i)
  z⁺ = A.z[:,i+1]
  z = A.z[:,i]
  u⁺ = A.u[:,i+1]
  u = A.u[:,i]
  Δt = A.Δt

  I = z .* (A.Δt - re)^3 /6Δt .+ z⁺ .* (re)^3 /6Δt
  C = (u⁺./Δt .- z⁺.*Δt/6).*(re)
  D = (u./Δt .- z.*Δt/6).*(A.Δt - re)
  I + C + D
end

function DataInterpolations.derivative(A::CubicSplineRegularGrid{<:CuArray{<:Number}}, t::Number)
    re = t%A.Δt
    i = floor(Int,t/A.Δt+ 1)
    i == i >= length(A.t) ? i = length(A.t) - 1 : nothing
    i == min(1,i)
    z⁺ = A.z[:,i+1]
    z = A.z[:,i]
    u⁺ = A.u[:,i+1]
    u = A.u[:,i]
    Δt = A.Δt

    dI = -3z .* (A.Δt - re)^2 /6Δt .+ 3z⁺ .* (re)^2 / 6Δt
    dC = u⁺./Δt    .- z⁺.*Δt/ 6
    dD = -(u./Δt .- z.*Δt/ 6)
    dI .+ dC .+ dD
end
TridiagonalGPUorCPU(a::Tridiagonal{T,N}) where{T,N<:CuVector} = convert(CuArray, a)

function DataInterpolations._interpolate(A::CubicSpline{<:CuMatrix,<:CuMatrix}, t::Number)
    i = min.(max.((sum(A.t.<t,dims=2)),1), sum(isfinite.(A.t), dims=2).-1) |> vec
    i⁺= vec(i.+1)
    i = CartesianIndex.(Base.OneTo(size(A.u,1)) |> cu, i)
    i⁺ = CartesianIndex.(Base.OneTo(size(A.u,1)) |> cu, i⁺)
    I = A.z[i] .* (A.t[i⁺] .- t).^3 ./ (6A.h[i⁺]) .+ A.z[i⁺] .* (t .- A.t[i]).^3 ./ (6A.h[i⁺])
    C = (A.u[i⁺]./A.h[i⁺] .- A.z[i⁺].*A.h[i⁺]./6).*(t .- A.t[i])
    D = (A.u[i]./A.h[i⁺] .- A.z[i].*A.h[i⁺]./6).*(A.t[i⁺] .- t)
    I .+ C .+ D
end