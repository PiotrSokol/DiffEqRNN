using Adapt
using Adapt: adapt_structure, adapt
using Zygote: @adjoint, @nograd


struct CheapEmbedding{T<:Integer, L, I<:AbstractVector{T}} <: AbstractMatrix{Bool}
  indices::I
end
CheapEmbedding{T, L, I}(indices) where {T, L, I} = CheapEmbedding{T, L, I}(indices)
CheapEmbedding(indices::AbstractVector{T}, L::Integer) where {T} = CheapEmbedding{T, L, typeof(indices)}(indices)
CheapEmbedding(indices::AbstractArray{T}, L::Integer) where {T} = CheapEmbedding(vec(indices), L) 
_indices(x::CheapEmbedding) = x.indices

Base.size(x::CheapEmbedding{<:Any, L}) where L = (Int(L), length(x.indices))
_onehotindex(x, i) = (x == i)
Base.getindex(x::CheapEmbedding, i::Integer, I...) = _onehotindex.(x.indices[I...], i)
Adapt.adapt_structure(T, x::CheapEmbedding{<:Any, L}) where L = CheapEmbedding(adapt(T, x.indices), L)
# Base.BroadcastStyle(::Type{<:CheapEmbedding}) where N = CUDA.CuArrayStyle{1}()

@nograd CheapEmbedding

function Base.:(*)(A::AbstractMatrix, B::CheapEmbedding{<:Any, L}) where L
  return A[:, B.indices]
end
@adjoint function Base.:(*)(A::AbstractMatrix, B::CheapEmbedding{<:Any, L}) where L
    c = A[:, B.indices]
    return c, Δ -> ( Δ*eltype(c).(B.indices .== (1:L)'), nothing )
end