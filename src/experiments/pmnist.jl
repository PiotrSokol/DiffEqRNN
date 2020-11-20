using DiffEqFlux, OrdinaryDiffEq, Flux, Optim
include("/Users/piotrsokol/Documents/RNNODE.jl/src/rnn_ode.jl")
import DiffEqSensitivity: ischeckpointing, isconcretetype
using Zygote
using MLDataUtils
using Flux: logitcrossentropy
using Flux.Data: DataLoader
using MLDatasets, NNlib, MLDataUtils
#ENV["PYTHON"] = "/Users/piotrsokol/anaconda3/envs/bortho/bin/python"
#using Pkg; Pkg.build("PyCall")
using PyCall

using CUDA
CUDA.allowscalar(false)
use_gpu = false

py"""
import sys
sys.path.insert(0, "/Users/piotrsokol/Documents/block-orthogonal/src/")
"""
py_seeds= pyimport("data_utils").__fixed_seeds__
seeded_rng = py_random.RandomState(py_seeds[set])
permutation = seeded_rng.permutation(1:784)

function loadmnist(batchsize = bs, train_split = 1. - 1/60)
    # Use MLDataUtils LabelEnc for natural onehot conversion
    onehot(labels_raw) = convertlabel(LabelEnc.OneOfK, labels_raw,
                                      LabelEnc.NativeLabels(collect(0:9)))
    # Load MNIST
    imgs, labels_raw = MNIST.traindata();
    # Process images into (H,W,C,BS) batches
    x_data = Float32.(reshape(imgs, size(imgs,1), size(imgs,2), 1, size(imgs,3)))
    y_data = onehot(labels_raw)
    (x_train, y_train), (x_test, y_test) = stratifiedobs((x_data, y_data),
                                                         p = train_split)
    return (
        # Use Flux's DataLoader to automatically minibatch and shuffle the data
        DataLoader(gpu.(collect.((x_train, y_train))); batchsize = batchsize,
                   shuffle = true),
        # Don't shuffle the test data
        DataLoader(gpu.(collect.((x_test, y_test))); batchsize = batchsize,
                   shuffle = false)
    )
end