using DiffEqFlux, OrdinaryDiffEq, Flux, Optim
include("/Users/piotrsokol/Documents/RNNODE.jl/src/rnn_ode.jl")
using Zygote
using MLDataUtils
using Flux: logitcrossentropy
using Flux.Data: DataLoader
using MLDatasets, NNlib, MLDataUtils
#ENV["PYTHON"] = "/Users/piotrsokol/anaconda3/envs/bortho/bin/python"
#using Pkg; Pkg.build("PyCall")
using PyCall
using CUDA
using Parameters: @with_kw, @unpack
"""
    Instead of using Argparse, use Paramters.@with_kw since it requires less compilation time
"""
@with_kw mutable struct Args
    _seed::Int
    alpha::Float = 1.5f0
    batchsize::Int = 128
    cuda::Bool=false
    data_dir = nothing
    dataset::String = "cm"
    factor = 1.f0
    gradient_clipping::Float = 0.f0
    hidden_size::Int = 250
    hpsearch::Bool=false
    initializer::String="default"; @assert initializer ∈ ["default","limitcycle"]
    input_size::Int = dataset == "cm" ? 9 : 10
    interpolation::String="PiecewiseConstant"; @assert interpolation=="PiecewiseConstant"
    lr = 0.01f0
    max_epochs::Int = dataset == "cm" ? 60 : 150
    max_lag::Int=120
    min_lag::Int=100
    optimizer::String = "ADAM"; @assert optimizer ∈ ["ADAM","SGD"]
    output_size::Int = 9
    python_code_dir = nothing
    save_dir = nothing
    time_interval::Int = 100
end




function onehot(labels_raw, ntoken::Int=9)
    return  convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:ntoken)))
end

function get_data(batchsize, dataset, device; max_lag=nothing, min_lag=nothing, set::String="train", time_interval=nothing)
    N = Dict(:train=> Int(1e4), :test=> Int(1e3), :valid=>Int(1e4))
    if dataset == "rcm"
        data_generator = pyimport("data_utils")["_generate_copy_memory"]
    elseif dataset == "cm"
        data_generator() = pyimport("data_utils")["_generate_random_copy_memory"](time_interval, )
    else
    x,y = data_generator
    
DataLoader(device.(x, y); batchsize = batchsize, shuffle = set == "train" ? true : false)
end

function get_network(;alpha, architecture, initializer, input_size, hidden_size,output_size, tsteps)
    if architecture == "RNN_TANH"
        ∂rnncell = ∂RNNCell
    elseif architecture == "GRU"
        ∂rnncell = ∂GRUCell
    else
        ∂rnncell = ∂LSTMCell
    
    ∂rnn = initializer == "limitcycle" || architecture == "LSTM" ? ∂rnncell(input_size, hidden_size) : ∂rnncell(input_size,hidden_size,initializer)

    node = RNNODE(∂rnn, (0.f0, tspan[end]), saveat=tsteps)
    function interpolate(x)
        x = Zygote.ignore() do
            ConstantInterpolationFixedGrid(x)
        end
    end
    return Chain( interpolate, node, Dense(hidden_size, output_size))
end
function experiment(; kwargs)
    @unpack batchsize, hidden_size, input_size, output_size, max_epochs, data_dir, save_dir, factor, lr, optimizer, hpsearch, gradient_clipping, alpha, cuda, python_code_dir, min_lag, max_lag, time_interval, dataset = kwargs

    tsteps =  dataset == "rcm" ? collect(Float32, 1:max_lag+2sequence_len+1) : collect(Float32, 1:time_interval+2sequence_len+1)
    get_network(alpha=alpha, architecture=architecture, initializer=initializer, input_size=input_size, hidden_size=hidden_size,output_size=output_size, tsteps)

    
    """
    unpack kwargs into some compact notation
    """
    py"""
    import sys
    sys.path.insert(0, $python_code_dir)
    """
    

end



if abspath(PROGRAM_FILE) == @__FILE__ 
    experiment()
end