# Cubic Spline Interpolation
struct CubicSplineRegularGrid{uType,tType,RangeType,zType,FT,T} <: AbstractInterpolation{FT,T}
  u::uType
  t₀::tType
  t₁::tType
  Δt::tType
  t::RangeType
  z::zType
  CubicSplineRegularGrid{FT}(u,t₀,t₁,Δt,z) where FT = new{typeof(u),eltype(u),typeof(t₀:Δt:t₁),typeof(z),FT,eltype(u)}(u,t₀,t₁,Δt,t₀:Δt:t₁,z)
end

function CubicSplineRegularGrid(u::AV; t₀::T=0,t₁::T=length(u)-1,Δt::T=1) where {T<:Number,AV<:AbstractVector{<:Number}}
  @assert ~any(ismissing, u)
  n = length(u) - 1
  @assert length(t₀:Δt:t₁-Δt) == n
  h = vcat( zero(eltype(u)), Δt*ones(eltype(u),n), zero(eltype(u)))
  dl = h[2:n+1]
  d_tmp = 2 .* (h[1:n+1] .+ h[2:n+2])
  du = h[2:n+1]
  tA = Tridiagonal(dl,d_tmp,du)
  d = map(i -> i == 1 || i == n + 1 ? 0 : 6(u[i+1] - 2u[i] + u[i-1])/Δt, 1:n+1)
  z = tA\d
  CubicSplineRegularGrid{true}(u,t₀,t₁,Δt,z)
end

function CubicSplineRegularGrid(U::AV; t₀::T=0,t₁::T=size(U,2)-1,Δt::T=1) where {T<:Number,AV<:AbstractMatrix{<:Number}}
  @assert ~any(ismissing, U)
  n = size(U,2) - 1
  @assert length(t₀:Δt:t₁-Δt) == n
  h = vcat( zeros(eltype(U)), Δt*ones(eltype(U),n), zeros(eltype(U)))
  du = dl = copy(h[2:n+1])
  d_tmp = 2 .* (h[1:n+1] .+ h[2:n+2])

  tA = Tridiagonal(dl,d_tmp,du)
  d = reduce(hcat, map(i -> i == 1 || i == n + 1 ? zeros(eltype(U), size(U,1)) : 6(view(U, :, i+1) - 2view(U, :, i) + view(U, :, i-1)  )/Δt, 1:n+1) )
  z = tA\permutedims(d) |>permutedims
  CubicSplineRegularGrid{true}(U,t₀,t₁,Δt,z)
end

function DataInterpolations._interpolate(A::CubicSplineRegularGrid{<:AbstractVector{<:Number}},t::Number)
  re = eltype(A.u)(t%A.Δt)
  i = floor(Int32,t/A.Δt + 1)
  i == i >= length(A.t) ? i = length(A.t) - 1 : nothing
  i == 0 ? i += 1 : nothing

  z(i) = A.z[i]
  u(i) = A.u[i]
  Δt = A.Δt
  I = @views z(i) .* (Δt - re)^3 /6Δt .+ z(i+1) .* (re)^3 /6Δt #check
  C = @views (u(i+1)/Δt .- z(i+1).*Δt/6).*(re)
  D = @views (u(i)/Δt .- z(i).*Δt/6).*(Δt- re)
  I + C + D
end

function DataInterpolations._interpolate(A::CubicSplineRegularGrid{<:AbstractMatrix{T}},t::Number) where {T<:Number}
interpolation = ignore() do
    interpolation = Vector{T}(undef,size(A.u,1))
    re = T(t%A.Δt)
    i = floor(Int32,t/A.Δt + 1)
    i == i >= length(A.t) ? i = length(A.t) - 1 : nothing
    i == 0 ? i += 1 : nothing

    u(i) = view(A.u, :,i)
    z(i) = view(A.z, :,i)
    Δt = A.Δt

    interpolation .= z(i) .* (Δt - re)^3 /6Δt .+ z(i+1) .* (re)^3 /6Δt
    interpolation .+= (u(i+1)/Δt .- z(i+1).*Δt/6).*(re)
    interpolation .+= (u(i)/Δt .- z(i).*Δt/6).*(Δt - re)

    interpolation
  end
end

struct LinearInterpolationRegularGrid{uType,tType,FT,T} <: AbstractInterpolation{FT,T}
  u::uType
  t::tType
  LinearInterpolationRegularGrid{FT}(u,t) where FT = new{typeof(u),typeof(t),FT,eltype(u)}(u,t)
end

function LinearInterpolationRegularGrid(u::AV; t₀::T=0,t₁::T=length(u)-1,Δt::T=1) where {T<:Number,AV<:AbstractVector{<:Number}}
    n = length(u)
    @assert length(t₀:Δt:t₁) == n
    t = collect(eltype(u),t₀:Δt:t₁)
    return LinearInterpolationRegularGrid{true}(u,t)
end

function LinearInterpolationRegularGrid(U::AV; t₀::T=0,t₁::T=size(U,2)-1,Δt::T=1) where {T<:Number,AV<:AbstractMatrix{<:Number}}
    n = size(U,2)
    @assert length(t₀:Δt:t₁) == n
    t = collect(eltype(U),t₀:Δt:t₁)
    return LinearInterpolationRegularGrid{true}(U,t)
end

function ConstantInterpolationRegularGrid(u::AV; t₀::T=0,t₁::T=length(u)-1,Δt::T=1) where {T<:Number,AV<:AbstractVector{<:Number}}
    n = length(u)
    @assert length(t₀:Δt:t₁) == n
    t = collect(t₀:Δt:t₁-Δt)
    return ConstantInterpolation{true}(u,t,:left)
end

function ConstantInterpolationRegularGrid(U::AV; t₀::T=0,t₁::T=size(U,2)-1,Δt::T=1) where {T<:Number,AV<:AbstractMatrix{<:Number}}
    n = size(U,2)
    @assert length(t₀:Δt:t₁) == n
    t = collect(t₀:Δt:t₁)
    return ConstantInterpolation{true}(U,t,:left)
end

function DataInterpolations._interpolate(A::LinearInterpolationRegularGrid{<:AbstractVector{<:Number}}, t::Number)
  idx = findfirst(x->x>=t,A.t)
  idx == nothing ? idx = length(A.t) - 1 : idx -= 1
  idx == 0 ? idx += 1 : nothing
  θ = (t - A.t[idx])/ (A.t[idx+1] - A.t[idx])
  
  @views (1-θ)*A.u[idx] + θ*A.u[idx+1]
end

function DataInterpolations._interpolate(A::LinearInterpolationRegularGrid{<:AbstractMatrix{<:Number}}, t::Number)
  idx = findfirst(x->x>=t,A.t)
  idx == nothing ? idx = length(A.t) - 1 : idx -= 1
  idx == 0 ? idx += 1 : nothing
  θ = (t - A.t[idx])/ (A.t[idx+1] - A.t[idx])
  @views (1-θ)*A.u[:,idx] + θ*A.u[:,idx+1]
end
"""
Modifying Flux.Data._getobs to work with (some) subtypes of AbstractInterpolation
"""
_getobs(data::ConstantInterpolation{<:AbstractMatrix}, i) = ConstantInterpolation{true}(data.u[i,:],data.t,data.dir)

_getobs(data::LinearInterpolationRegularGrid{<:AbstractMatrix}, i) = LinearInterpolation{true}(data.u[i,:],data.t)
_getobs(data::CubicSplineRegularGrid{<:AbstractMatrix}, i) = CubicSplineRegularGrid{true}(data.u[i,:], data.t.start, data.t.stop, data.t.step, data.z[i,:])
_getobs(data::CubicSpline{<:AbstractMatrix}, i) = CubicSpline{true}(data.u[i,:], data.t, data.h, data.z[i,:])
_nobs(n::T) where {T<:Union{ConstantInterpolation,LinearInterpolationRegularGrid,CubicSplineRegularGrid,CubicSpline}} =  size(n.u,1)



function DataInterpolations.derivative(A::LinearInterpolationRegularGrid{<:AbstractVector{<:Number}}, t::Number)
    idx = findfirst(x -> x >= t, A.t) - 1
    idx == 0 ? idx += 1 : nothing
    θ = 1 / (A.t[idx+1] - A.t[idx])
    @views (A.u[idx+1] - A.u[idx]) / (A.t[idx+1] - A.t[idx])
end

function DataInterpolations.derivative(A::LinearInterpolationRegularGrid{<:AbstractMatrix{<:Number}}, t::Number)
    idx = findfirst(x -> x >= t, A.t) - 1
    idx == 0 ? idx += 1 : nothing
    θ = 1 / (A.t[idx+1] - A.t[idx])
    @views (A.u[:, idx+1] - A.u[:, idx]) / (A.t[idx+1] - A.t[idx])
end
"""
  Implicitly the derivatives for CubicSplines assume a constant extrapolation.
"""
function DataInterpolations.derivative(C::CubicSplineRegularGrid{<:AbstractVector{<:Number}}, t::Number)
    re = t%C.Δt
    i = floor(Int,t/C.Δt + 1)
    i == i >= length(B.t) ? i = length(B.t) - 1 : nothing
    i == min(1,i)
    dI = -3C.z[i] * (C.t[i + 1] - t)^2 /6C.Δt + 3C.z[i + 1] * (t - C.t[i])^2 /6C.Δt
    dC = C.u[i + 1]/C.Δt - C.z[i + 1] * C.Δt / 6
    dD = -(C.u[i]/C.Δt - C.z[i] * C.Δt / 6)
    dI + dC + dD
end

function DataInterpolations.derivative(B::CubicSplineRegularGrid{<:AbstractMatrix{<:T}}, t::Number) where {T<:Number}
interpolation = ignore() do
    interpolation = Vector{T}(undef,size(B.u,1))
    re = t%B.Δt
    i = floor(Int,t/B.Δt + 1)
    i == i >= length(B.t) ? i = length(B.t) - 1 : nothing
    i == min(1,i)
    u(i) = view(B.u, :,i)
    z(i) = view(B.z, :,i)
    interpolation .= -3z(i) .* (B.Δt - re)^2 /6B.Δt .+ 3z(i+1) .* (re)^2 / 6B.Δt
    interpolation .+= u(i+1)/B.Δt .- z(i+1).*B.Δt/ 6
    interpolation .+= -(u(i)/B.Δt .- z(i).*B.Δt/6)
end
end
# Extending DataInterpolations.CubicSpline to work with 2-D arrays of inputs

"""
TridiagonalGPUorCPU returns dense matrix if the elements of Tridiagonal are CUDA vectors, otherwise is identity
"""
TridiagonalGPUorCPU(a::Tridiagonal{T,N}) where {T,N<:AbstractVector} = a

function DataInterpolations.CubicSpline(u::AbstractMatrix{<:Number},t::AbstractVector{<:Number})
    t = reshape(t,1,:)
    @assert length(t) == size(u, 2)
    n = size(t,2) - 1
    h = similar(t, Base.Dims{1}(n+2))
    fill!(h, eltype(t)(0))
    Δt = diff(t, dims=2)
    h[2:end-1].= Δt[1,:]
    dl = h[2:n+1]
    d_tmp = 2 .* (h[1:n+1] .+ h[2:n+2])
    tA = Tridiagonal(dl,d_tmp, dl)
    tA = TridiagonalGPUorCPU(tA)
    d = zero(u)
    # fill!(d,eltype(u|>eltype)(0))
    d[:,2:end-1].= @views 6diff(u,dims=2)[:,2:end]./Δt[:,2:end] - 6diff(u,dims=2)[:,1:end-1]./Δt[:,1:end-1]
    z = permutedims(tA\permutedims(d))
    CubicSpline{true}(u,vec(t),reshape(h[1:n+1],1,:),z)
end

function DataInterpolations.CubicSpline(u::U,t::T) where {U<:Vector{<:AbstractVector{<:Number}},T<:Vector{<:AbstractVector{<:Number}}}
    @assert length.(t) == length.(u)
    n = length.(t) .-1
    h = similar.(t, Base.Dims{1}.(n.+2))
    FT = eltype(t|>eltype)

    fill!.(h, FT(0))
    Δt = diff.(t)
    copyto!.(h, 2, Δt, 1, length.(Δt))
    dl = getindex.(h,UnitRange{Int}.(Ref(2),n.+1))
    d_tmp =2(Base.getindex.(h,UnitRange{Int}.(Ref(1),n.+1)) .+ Base.getindex.(h,UnitRange{Int}.(Ref(2),n.+2)))
    tA = Tridiagonal.(dl,d_tmp, dl)
    tA = TridiagonalGPUorCPU.(tA)
    d = zero.(u)
    left(A) = Base.getindex.(A,UnitRange{Int}.(Ref(2),n))
    right(A) = Base.getindex.(A,UnitRange{Int}.(Ref(1),n.-1))
    tmp_left = 6left(diff.(u))
    tmp_right = 6right(diff.(u));
    tmp = [tmp_left[i]./Δt[i][2:n[i]] .- tmp_right[i]./Δt[i][1:n[i]-1] for i ∈ 1:length(n)]
    copyto!.(d,2, tmp, 1, length.(tmp));
    z=broadcast(\, tA, d)
    maxpad = maximum(n).+1
    h = Base.getindex.(h, UnitRange{Int}.(Ref(1),n.+1))

    u = reduce(hcat, rpad.(u,maxpad,FT(Inf))) |> permutedims
    t = reduce(hcat, rpad.(t,maxpad,FT(Inf))) |> permutedims
    h = reduce(hcat, rpad.(h,maxpad,FT(Inf))) |> permutedims
    z = reduce(hcat, rpad.(z,maxpad,FT(Inf))) |> permutedims
    CubicSpline{true}(u,t,h,z)
end
"""
  Interpolation call for a CubicSpline with a vector for time series for each
  row of data points ('batched' interpolation)
"""
function DataInterpolations._interpolate(A::DataInterpolations.CubicSpline{<:AbstractMatrix,<:AbstractMatrix}, t::Number)
    i = min.(max.((sum(A.t.<t,dims=2)),1), sum(isfinite.(A.t), dims=2).-1) |> vec
    i⁺= vec(i.+1)
    i = CartesianIndex.(Base.OneTo(size(A.u,1)), i)
    i⁺ = CartesianIndex.(Base.OneTo(size(A.u,1)), i⁺)
    I = @views A.z[i] .* (A.t[i⁺] .- t).^3 ./ (6A.h[i⁺]) .+ A.z[i⁺] .* (t .- A.t[i]).^3 ./ (6A.h[i⁺])
    C = @views (A.u[i⁺]./A.h[i⁺] .- A.z[i⁺].*A.h[i⁺]./6).*(t .- A.t[i])
    D = @views (A.u[i]./A.h[i⁺] .- A.z[i].*A.h[i⁺]./6).*(A.t[i⁺] .- t)
    I .+ C .+ D
end
"""
  Interpolation call for a CubicSpline with a single vector of time points
"""
function DataInterpolations._interpolate(A::DataInterpolations.CubicSpline{<:AbstractMatrix,<:AbstractVector},t::Number)
    T = reshape(A.t,1,:)
    i = min.(max.( sum(T.<t,dims=2),1), length(T).-1) |> vec
    i⁺= vec(i.+1)
    I = @views A.z[:,i] .* (T[:,i⁺] .- t).^3 ./ (6A.h[:,i⁺]) .+ A.z[:,i⁺] .* (t .- T[:,i⁺]).^3 ./ (6A.h[:,i⁺])
    C = @views (A.u[:,i⁺]./A.h[:,i⁺] .- A.z[:,i⁺].*A.h[:,i⁺]./6).*(t .- T[:,i])
    D = @views (A.u[:,i]./A.h[:,i⁺] .- A.z[:,i].*A.h[:,i⁺]./6).*(T[:,i⁺] .- t)
    I .+ C .+ D
end

# function DataInterpolations.derivative(A::DataInterpolations.CubicSpline{<:AbstractVector{<:Number}}, t::Number)
#     i = findfirst(x -> x >= t, A.t)
#     i == nothing ? i = length(A.t) - 1 : i -= 1
#     i == 0 ? i += 1 : nothing
#     dI = -3A.z[i] * (A.t[i + 1] - t)^2 / (6A.h[i + 1]) + 3A.z[i + 1] * (t - A.t[i])^2 / (6A.h[i + 1])
#     dC = A.u[i + 1] / A.h[i + 1] - A.z[i + 1] * A.h[i + 1] / 6
#     dD = -(A.u[i] / A.h[i + 1] - A.z[i] * A.h[i + 1] / 6)
#     dI + dC + dD
# end
"""
  Derivative method for a CubicSpline with a vector for time series for each
  row of data points
"""
function DataInterpolations.derivative(A::DataInterpolations.CubicSpline{<:AbstractMatrix,<:AbstractMatrix}, t::Number)
    i = min.(max.((sum(A.t.<t,dims=2)),1), sum(isfinite.(A.t), dims=2).-1) |> vec
    i⁺= vec(i.+1)
    i = CartesianIndex.(Base.OneTo(size(A.u,1)), i)
    i⁺ = CartesianIndex.(Base.OneTo(size(A.u,1)), i⁺)
    dI = @views -3 .*A.z[i] .* (A.t[i⁺] .- t).^2 ./ 6A.h[i⁺] .+ 3 .*A.z[i⁺] .* (t .- A.t[i]).^2 ./ 6A.h[i⁺]
    dC = @views (A.u[i⁺]./A.h[i⁺] .- A.z[i⁺].*A.h[i⁺]./6)
    dD = @views -(A.u[i]./A.h[i⁺] .- A.z[i].*A.h[i⁺]./6)
    dI .+ dC .+ dD
end
"""
  Derivative method for a CubicSpline with a single vector of time points
"""
function DataInterpolations.derivative(A::DataInterpolations.CubicSpline{<:AbstractMatrix,<:AbstractVector},t::Number)
    T = reshape(A.t,1,:)
    i = min.(max.( sum(T.<t,dims=2),1), length(T).-1) |> vec
    i⁺= vec(i.+1)
    dI = @views -3 .* A.z[:,i] .* (T[:,i⁺] .- t).^2 ./ (6A.h[:,i⁺]) .+ 3 .* A.z[:,i⁺] .* (t .- T[:,i⁺]).^2 ./ (6A.h[:,i⁺])
    dC = @views (A.u[:,i⁺]./A.h[:,i⁺] .- A.z[:,i⁺].*A.h[:,i⁺]./6)
    dD = @views -(A.u[:,i]./A.h[:,i⁺] .- A.z[:,i].*A.h[:,i⁺]./6)
    dI .+ dC .+ dD
end
