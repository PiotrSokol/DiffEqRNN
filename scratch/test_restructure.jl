using Flux, Zygote
using Zygote:@adjoint, Buffer
function destructure(m; cache = IdDict())
  xs = Buffer([])
  fmap(m) do x
    if x isa AbstractArray
      push!(xs, x)
    else
      cache[x] = x
    end
    return x
  end
  return vcat(vec.(copy(xs))...), p -> _restructure(m, p, cache = cache)
end

function _restructure(m, xs; cache = IdDict())
  i = 0
  fmap(m) do x
    x isa AbstractArray || return cache[x]
    x = reshape(xs[i.+(1:length(x))], size(x))
    i += length(x)
    return x
  end
end
@adjoint function _restructure(m, xs; cache = IdDict())
  _restructure(m, xs, cache = cache), dm -> (nothing,destructure(dm, cache = cache)[1])
end

rnn = Flux.RNNCell(10, 10, tanh)
p, re = destructure(rnn)
∇ = let u0 = randn(Float32, 10,17), x = randn(Float32, 10,17)
  print(p)
  ∇ = Zygote.gradient( p-> sum(abs2, re(p)(u0,x)[2]), p)
end
