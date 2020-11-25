using DiffEqFlux, OrdinaryDiffEq, Flux, Optim
include("/Users/piotrsokol/Documents/RNNODE.jl/src/rnn_ode.jl")
using Zygote
using Flux: logitcrossentropy
using Flux.Data: DataLoader
using MLDatasets, NNlib, MLDataUtils
#ENV["PYTHON"] = "/Users/piotrsokol/anaconda3/envs/bortho/bin/python"
#using Pkg; Pkg.build("PyCall")
using PyCall
using CUDA
using Parameters: @with_kw, @unpack
import Statistics: mean
using BSON,NPZ
using UUIDs
using EllipsisNotation
using ProgressMeter
using ArgParse
FT = Float32
"""
    Instead of using Argparse, use Paramters.@with_kw since it requires less compilation time
"""
# @with_kw mutable struct Args
#     _seed::Int
#     alpha::Float32 = 1.5f0
#     architecture::String = "RNN_TANH"; @assert architecture ∈ ["RNN_TANH", "GRU", "LSTM"]
#     batchsize::Int = 128
#     cuda::Bool=false
#     data_dir = nothing
#     dataset::String = "cm"
#     factor = 1.f0
#     gradient_clipping::Float32 = 0.f0
#     hidden_size::Int = 250
#     hpsearch::Bool=false
#     initializer::String="default"; @assert initializer ∈ ["default","limitcycle"]
#     input_size::Int = 10
#     interpolation::String="PiecewiseConstant"; @assert interpolation=="PiecewiseConstant"
#     lr::Float32 = 0.01f0
#     max_epochs::Int = dataset == "cm" ? 60 : 150
#     max_lag::Int=120
#     min_lag::Int=100
#     optimizer::String = "ADAM"; @assert optimizer ∈ ["ADAM","Momentum"]
#     output_size::Int = 9
#     patience::Int=10
#     python_code_dir = nothing
#     save_dir = pwd()
#     save_name::String
#     sequence_len = 10
#     time_interval::Int = 100
# end

function onehot(labels_raw; ntoken::Int=9)
    return  convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:ntoken)))
end

function get_data(batchsize, dataset, device; max_lag, min_lag, set::Symbol, time_interval)
    
    N = Dict(:train=> Int(1e4), :test=> Int(1e3), :valid=>Int(1e4))
    if dataset == "rcm"
        x,y = pyimport("data_utils")["_generate_random_copy_memory"](min_lag,max_lag,N[set] ,set)
    elseif dataset == "cm"
        x,y = pyimport("data_utils")["_generate_copy_memory"](time_interval, N[set], set)
        
    end
    
    x = permutedims(x)
    y = reshape(y, size(y)[1],size(y)[2],1)
    y = mapslices( x-> onehot(vec(x)), y, dims=[1,3])

    return DataLoader(device.((x, y)); batchsize = batchsize, shuffle = set == :train ? true : false)
end

function get_loss(dataset, max_lag, min_lag, time_interval)
    
    blank_token = 8
    weight = ones(FT, output_size)
    weight[blank_token] /= dataset == "cm" ? time_interval : mean([min_lag, max_lag]) 
    function logitcrossentropy(ŷ, y; dims=1, agg=mean)
    agg(.-sum( (y .* logsoftmax(ŷ; dims=dims)).* weight; dims=dims))
    end
    return logitcrossentropy
end


function get_network(alpha, architecture, initializer, input_size, hidden_size,output_size, tsteps)
    if architecture == "RNN_TANH"
        ∂rnncell = ∂RNNCell
    elseif architecture == "GRU"
        ∂rnncell = ∂GRUCell
    else
        ∂rnncell = ∂LSTMCell
    end
    """
    Ternary op -> reads as if initializer == "limitcycle" or architecture == "LSTM" use two argument function dispatch, else additionally pass initializer variable
    """
    ∂rnn = initializer == "limitcycle" || architecture == "LSTM" ? ∂rnncell(input_size, hidden_size) : ∂rnncell(input_size,hidden_size,Flux.glorot_uniform)

    node = RNNODE(∂rnn, (0.f0, tsteps[end]), saveat=tsteps, preprocess=x-> Float32.(onehot(x, ntoken=input_size-1)) )

    function interpolate(x)
        X = Zygote.ignore() do
            permutedims(x) |> ConstantInterpolationFixedGrid
        end
    end
    return Chain( interpolate, node, Array, x-> reshape(x, hidden_size, prod(size(x)[2:3])), Dense(hidden_size, output_size) )
end

classify(x) = argmax.(eachcol(x))

function evaluate_set(model, data, 𝓁array, accarray, ℒ, set)
    loss_set = Float32[]
    total_correct = 0
    total = 0
    @showprogress "Evaluating $set "  for (x,y) in data
        # Only evaluate accuracy for n_batches
        y = reshape(permutedims(y, (1,3,2)),9, prod(size(y)[2:3]))
        ŷ = model(x)
        𝓁 = ℒ(ŷ,y)
        push!(loss_set, 𝓁[1])
        target_class = classify(cpu(y))
        predicted_class = classify(cpu(ŷ))
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    push!(accarray, (total_correct / total)[1] )
    push!(𝓁array, mean(loss_set) )
    return nothing
end

function experiment(args="" )

    s = ArgParseSettings()
    s.prog = "cm_rcm.jl : Copy-Memmory / Random Copy-Memmory "

    s.description="Trains a continous time RNN on the copy memory task, with potentially random delay between the input sequence and output."

    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        "--_seed"
            arg_type = Int
            default = 1
        "--alpha"
            arg_type = Float32
            default = 1.5f0
            # dest_name = α
        "--architecture"
            arg_type = String
            default = "RNN_TANH"
            range_tester = (x->x ∈ ["RNN_TANH", "GRU", "LSTM"])
        "--bs"
            arg_type = Int
            default = 100
        "--cuda"
            action = :store_true
        "--data_dir"
            arg_type = String
            default = pwd()
        "--dataset"
            arg_type = String
            default = "cm"
        "--factor"
            arg_type = Float32
            default = 1.f0
            range_tester = (x->  0. < x <= 1. )
        "--gradient_clipping"
            arg_type = Float32
            default = Float32(1e3)
            range_tester = (x->  0. < x)
        "--hidden_size"
            arg_type = Int
            default = 250
        "--hpsearch"
            action = :store_true
        "--initializer"
            arg_type = String
            default = "limitcycle"
            range_tester = (x->x ∈ ["limitcycle", "default", "eoc"])
        "--input_size"
            arg_type = Int
            default = 10
        "--interpolation"
            arg_type = String
            default = "PiecewiseConstant"
            range_tester = (x -> x ∈ ["PiecewiseConstant"])
        "--lr"
            arg_type = Float32
            default = Float32(1e-3)
            # dest_name = η
        "--max_epochs"
            arg_type = Int
            default = 3 # TODO set to 60
        "--max_lag"
            arg_type = Int
            default = 120
        "--min_lag"
            arg_type = Int
            default = 100
        "--optimizer"
            arg_type = String
            default = "ADAM"
            # range_tester = (x->x ∈ ["ADAM", "Momentum"])
        "--output_size"
            arg_type = Int
            default = 9
        "--patience"
            arg_type = Int
            default = 10
        "--python_code_dir"
            arg_type = String
            default = "/Users/piotrsokol/Documents/block-orthogonal/src/"
            # required = true # TODO remove default, make required
        "--save_dir"
            arg_type = String
            default = pwd()
        "--save_name"
            arg_type = String
            default = string(uuid1())
        "--sequence_len"
            arg_type = Int
            default = 10
        "--time_interval"
            arg_type = Int
            default = 100
    end
    parsed = parse_args(s;as_symbols=true)
    for (k,v) in parsed
        myexpression = k == :optimizer ? :(OPT = $v) : :($(k) = $v)
        println(myexpression)
        eval(myexpression)
    end
    α = alpha
    η = lr
    # args = Args(; kw...)
    # @unpack architecture, batchsize, hidden_size, input_size, output_size, max_epochs, data_dir, save_dir, factor, optimizer, hpsearch, gradient_clipping, alpha, cuda, python_code_dir, patience, min_lag, max_lag, time_interval, dataset, initializer, save_name = args
    
    optimizer = Symbol(OPT)

    if cuda
        try 
            device = has_cuda ? gpu : cpu
            CUDA.allowscalar(false)
        catch ex
            @warn "CUDA requested but CUDA.jl fails to load" exception=(ex,catch_backtrace())
            device = cpu
        end
    else
        device = cpu
    end

    sets = hpsearch ? [:train, :valid] : [:train, :valid,:test]  # if doing hyperparameter search only get two data loaders, else get all 3

    """
    Get data
    
    1. Append python path with module that generates the data
    2. Get data for specific set ∈ :train, :valid, :test
    """

    py"""
    import sys
    sys.path.insert(0, $python_code_dir)
    """
    _get_data(set) = get_data(bs, dataset, device, max_lag=max_lag, min_lag=min_lag, set=set, time_interval=time_interval)
    if hpsearch
        train_loader,valid_loader = _get_data.(sets)
        eval_sets = Dict(:valid=>valid_loader)
    else
        train_loader,valid_loader, test_loader = _get_data.(sets)
        eval_sets = Dict(:valid=>valid_loader, :test=>test_loader)
        fname = string(uuid1(),".bson")
    end
    metrics = Dict(:train=> Dict(:loss=>FT[], :accuracy=>FT[]), :test=>Dict(:loss=>FT[], :accuracy=>FT[]), :valid=>Dict(:loss=>FT[], :accuracy=>FT[]))
    """
    Get network

    1. Define when the network output ought to be captured.
        *NB* To mimic the integration duration of a discrete-time RNN, we stagger the output times relative to the input times by Δt = 1.
    2. Get network from parameters
        *NB* alpha currently has no effect on the parameter initialization.
        # TODO: Fix α later! 
    """

    tsteps =  dataset == "rcm" ? collect(FT, 1:max_lag+2sequence_len) : collect(FT, 1:time_interval+2sequence_len)
    
    nn = get_network(α, architecture, initializer, input_size, hidden_size,output_size, tsteps)

    """
    Define the loss

    In the discrete time system (RNN) we used a re-weighted cross entropy; which down-weighed the blank token that the model has to output before and after it's required to output the sequence.

    The token is encoded as 8.
    """
    nll =  get_loss(dataset, max_lag, min_lag, time_interval)
    ℒ(ŷ::VecOrMat,y::VecOrMat) = nll(ŷ,y)
    ℒ(ŷ::VecOrMat,y::AbstractArray) = ℒ(ŷ, reshape(permutedims(y, (1,3,2)),9, prod(size(y)[2:3])))
    
    opt = Flux.Optimiser(ClipValue(gradient_clipping), eval(optimizer)(η))
    ΔIT = 0
    min_ℒ = Inf
    Δthr = 1e-4
    @showprogress "Epochs " for i in 1:max_epochs
        Flux.train!((x,y)->ℒ(nn(x),y), params(nn), train_loader, opt)
        
        if mapreduce(x -> isnan.(vec(x)), any ∘ vcat, params(nn))
            @error "NaN parameters detected. Breaking training loop."
            break
        end
        for set in sets if set!= :train
            evaluate_set(nn, eval_sets[set], metrics[set][:loss], metrics[set][:accuracy], set)
        end
        end

        if last(metrics[:valid][:loss]) < (1 - Δthr)min_ℒ
            ΔIT=0
            min_ℒ = last(metrics[:valid][:loss])
            if !hpsearch
                BSON.@save joinpath(save_dir, fname) params=cpu.(params(nn)) opt
            end
        elseif ΔIT >= patience && opt[2].eta > 1e-6
            opt[2].eta*= factor
        else
            ΔIT+=1
        end
    end
    if hpsearch
        metrics[:result] = min_ℒ
    else
        metrics[:fname] = fname
    end
    npzwrite(save_name, metrics)
end



if abspath(PROGRAM_FILE) == @__FILE__ 
    experiment(ARGS)
end