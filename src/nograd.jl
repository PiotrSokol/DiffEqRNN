const UnivInpt = Union{LinearInterpolationRegularGrid{T},ConstantInterpolation{T},CubicSpline{T},CubicSplineRegularGrid{T}} where  T<:AbstractVector
const MultivInpt = Union{LinearInterpolationRegularGrid{T},ConstantInterpolation{T},CubicSpline{T},CubicSplineRegularGrid{T}} where  T<:AbstractMatrix
"""
Wraps internally the inteprolation used in NeuralCDE's and RNNODE
in a Zygote.ignore block
"""
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

function derivative(n::nograd{<:UnivInpt}, t)
 x = ignore() do
    derivative(n.interpolant, t) |> n.f
  end
end

function derivative(n::nograd{<:MultivInpt}, t)
 x = ignore() do
    derivative(n.interpolant, t) |> n.f
  end
end

function infer_batchsizes(n::nograd)
   x = ignore() do
    size(n(0.1))[end]
  end
end
