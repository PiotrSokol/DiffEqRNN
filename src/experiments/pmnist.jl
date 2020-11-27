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
# using Parameters: @with_kw, @unpack
import Statistics: mean
using BSON,NPZ
using UUIDs
using EllipsisNotation
using ProgressMeter
using ArgParse
FT = Float32

function onehot(labels_raw; ntoken::Int=9)
    return  convertlabel(LabelEnc.OneOfK, labels_raw, LabelEnc.NativeLabels(collect(0:ntoken)))
end

function preprocess_img(imgs, p)
    imgs = reshape( imgs, prod(size(imgs)[1:2]), size(imgs)[3] ) |> x-> FT.(x)
    imgs = map(x-> permute!(x,p), eachcol(imgs)) |> x-> hcat(x...)
    imgs .-= 0.13066047
    imgs./= 0.30810785
    return imgs
end
function get_data(batchsize, device, set::Symbol; train_split=58999/60000)
    py_seeds= pyimport("data_utils").__fixed_seeds__
    py_random=pyimport("numpy.random");
    seeded_rng = py_random.RandomState(py_seeds["image"])
    p = seeded_rng.permutation(1:784)

    X, labels_raw = MNIST.traindata()
    Y = onehot(labels_raw)
    X = preprocess_img(X, p)

    (x_train, y_train), (x_valid, y_valid) = stratifiedobs((X, Y),p = train_split)
    
    if set!= :test
        return (
                DataLoader(device.(collect.((x_train, y_train))); batchsize = batchsize, shuffle = true),
                DataLoader(device.(collect.((x_valid, y_valid))); batchsize = batchsize, shuffle = false) )
    elseif set == :test
        X, labels_raw = MNIST.testdata()
        Y = onehot(labels_raw)
        X = preprocess_img(X, p)
        return (
                DataLoader(device.(collect.((x_train, y_train))); batchsize = batchsize, shuffle = true),
                DataLoader(device.(collect.((x_valid, y_valid))); batchsize = batchsize, shuffle = false),
                DataLoader(device.(collect.((X, Y))); batchsize = batchsize, shuffle = false)
            )
    end
end

function get_network(alpha, architecture, initializer, isize, hsize,osize, tsteps, interpolation)
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
    ∂rnn = initializer == "limitcycle" || architecture == "LSTM" ? ∂rnncell(isize, hsize) : ∂rnncell(isize,hsize,Flux.glorot_uniform)

    node = RNNODE(∂rnn, (0.f0, tsteps[end]), preprocess=x-> FT.(permutedims(x)), save_end=true, save_start=false, saveat=collect(0.f0:tsteps[end]) )

    function interpolate(x)
        X = Zygote.ignore() do
            permutedims(x) |> interpolation
        end
    end
    return Chain( interpolate, node, x->x(tsteps[end]), Dense(hsize, osize) )
end

classify(x) = argmax.(eachcol(x))

function evaluate_set(model, data, 𝓁array, accarray, ℒ, set)
    loss_set = Float32[]
    total_correct = 0
    total = 0
    @showprogress "Evaluating $set "  for (x,y) in data
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
    s.prog = "pmnist.jl : permuted sequential MNIST"

    s.description="Trains a continous time RNN on the permuted sequential MNIST task."

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
            default = "rcm"
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
            default = 512
        "--hpsearch"
            action = :store_true
        "--initializer"
            arg_type = String
            default = "limitcycle"
            range_tester = (x->x ∈ ["limitcycle", "default", "eoc"])
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
            default = 1 # TODO set to 60
        "--optimizer"
            arg_type = String
            default = "ADAM"
            # range_tester = (x->x ∈ ["ADAM", "Momentum"])
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
            default = "data"
        "--time_interval"
            arg_type = Int
            default = 100
    end
    parsed = parse_args(s;as_symbols=true)
    for (k,v) in parsed
        myexpression = k == :optimizer ? :(OPT = $v) : :($(k) = $v)
        eval(myexpression)
    end
    α = alpha
    η = lr
    optimizer = Symbol(OPT)
    Random.seed!(_seed)

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

    sets = hpsearch ? [:valid] : [:valid,:test]  # if doing hyperparameter search only get two data loaders, else get all 3

    """
    Get data
    
    1. Append python path with module that generates the data
    2. Get data for specific set ∈ :train, :valid, :test
    """

    py"""
    import sys
    sys.path.insert(0, $python_code_dir)
    """
    
    if hpsearch
        train_loader,valid_loader = get_data(bs, device, :valid)
        eval_sets = Dict(:valid=>valid_loader)
    else
        train_loader,valid_loader, test_loader = get_data(bs, device, :train)
        eval_sets = Dict(:valid=>valid_loader, :test=>test_loader)
        fname = save_name*".bson"
    end
    metrics = Dict(:test=>Dict(:loss=>FT[], :accuracy=>FT[]), :valid=>Dict(:loss=>FT[], :accuracy=>FT[]))
    """
    Get network

    1. Define when the network output ought to be captured.
        *NB* To mimic the integration duration of a discrete-time RNN, we stagger the output times relative to the input times by Δt = 1.
    2. Get network from parameters
        *NB* alpha currently has no effect on the parameter initialization.
        # TODO: Fix α later! 
    """

    tsteps =  FT(784)
    
    nn = get_network(alpha, architecture, initializer, isize, hsize,osize, tsteps, eval(Symbol(interpolation)))
    ℒ(ŷ,y) = logitcrossentropy(ŷ,y)
    
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
            evaluate_set(nn, eval_sets[set], metrics[set][:loss], metrics[set][:accuracy], ℒ, set)
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
        npzwrite(save_name*".npz", Dict("result"=>min_ℒ,
        "valid_loss"=>metrics[:valid][:loss],"valid_acc"=>metrics[:valid][:accuracy]))
    else
        npzwrite(save_name*".npz", Dict("valid_loss"=>metrics[:valid][:loss],"valid_acc"=>metrics[:valid][:accuracy],
        "test_loss"=>metrics[:test][:loss],"test_acc"=>metrics[:test][:accuracy]))
    end
    
    return 
end



if abspath(PROGRAM_FILE) == @__FILE__ 
    experiment(ARGS)
end