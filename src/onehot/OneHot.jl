module OneHot
    using NNlib
    using ChainRulesCore
    using Base.Broadcast: broadcasted
    using Statistics: mean
    const IntOrTuple = Union{Integer,Tuple}
    using Statistics: mean

    include("utils.jl")
    include("gather.jl")
    include("scatter.jl")
end