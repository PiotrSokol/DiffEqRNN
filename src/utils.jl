using Random
import LinearAlgebra.qr
##
function limit_cycle(rng::AbstractRNG, dims...;σ=1.5f0, θ₁=π/21,θ₂=π/3.8)
    @assert isequal(dims[1], dims[2]) && iseven(dims[1]) && iseven(dims[2])
    W = zeros(Float32, dims...)
    rand_block = θ ->  [cos(θ) -sin(θ) ; sin(θ) cos(θ)]
    for i in range(1, dims[1], step=2)
        W[i:i+1, i:i+1] = Float32(σ)*rand_block( Float32(θ₁)+rand(rng,Float32)*Float32(θ₂ - θ₁))
    end
    return W
end
## 
limit_cycle(dims...;kwargs...) = limit_cycle(Random.GLOBAL_RNG, dims...; kwargs...)
##
limit_cycle(rng::AbstractRNG; kwargs...) = (dims...; kwargs...) -> limit_cycle(rng, dims...; kwargs...)
##
function orthogonal_init(rng::AbstractRNG, dims...; σ=1.f0)
    Q,R = qr(randn(rng, Float32, dims...))
    return Float32(σ).*Q*sign.(Diagonal(R))
end
##
orthogonal_init(dims...;kwargs...) = orthogonal_init(Random.GLOBAL_RNG, dims...; kwargs...)
orthogonal_init(rng::AbstractRNG; kwargs...) = (dims...; kwargs...) -> orthogonal_init(rng, dims...; kwargs...)
