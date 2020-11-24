### A Pluto.jl notebook ###
# v0.12.11

using Markdown
using InteractiveUtils

# ╔═╡ cc458694-2dd1-11eb-1fba-6f838df9dcf2
using DiffEqFlux, OrdinaryDiffEq, Flux, Optim

# ╔═╡ e142e2b2-2dd1-11eb-0c98-311f697821e5
using Zygote

# ╔═╡ e14374f2-2dd1-11eb-21be-8b98b2880bd1
using Flux: logitcrossentropy

# ╔═╡ e1496202-2dd1-11eb-1ba8-af29cd392cde
using Flux.Data: DataLoader

# ╔═╡ e14999cc-2dd1-11eb-3993-fb0fec9916d3
using MLDatasets, NNlib, MLDataUtils

# ╔═╡ e14f5b32-2dd1-11eb-2275-27380c4af131
#ENV["PYTHON"] = "/Users/piotrsokol/anaconda3/envs/bortho/bin/python"
#using Pkg; Pkg.build("PyCall")
using PyCall

# ╔═╡ e15270ba-2dd1-11eb-0c92-a52dd7c668e6
using CUDA

# ╔═╡ e155f280-2dd1-11eb-3e21-4b7b624c5b91
using Parameters: @with_kw, @unpack

# ╔═╡ e15c4004-2dd1-11eb-0803-81cdc172c069
using NPZ

# ╔═╡ e16045dc-2dd1-11eb-36da-bf2c77219ab8
using UUIDs

# ╔═╡ e1110b02-2dd1-11eb-2a88-09037e941cf4
include("/Users/piotrsokol/Documents/RNNODE.jl/src/rnn_ode.jl")

# ╔═╡ e15621ba-2dd1-11eb-2aa3-77152d5a0477
import Statistics: mean

# ╔═╡ f5bd4480-2dd1-11eb-0535-ffba26a30be2


# ╔═╡ f588c5e8-2dd1-11eb-1859-67bb88f8fbed


# ╔═╡ Cell order:
# ╠═cc458694-2dd1-11eb-1fba-6f838df9dcf2
# ╠═e1110b02-2dd1-11eb-2a88-09037e941cf4
# ╠═e142e2b2-2dd1-11eb-0c98-311f697821e5
# ╠═e14374f2-2dd1-11eb-21be-8b98b2880bd1
# ╠═e1496202-2dd1-11eb-1ba8-af29cd392cde
# ╠═e14999cc-2dd1-11eb-3993-fb0fec9916d3
# ╠═e14f5b32-2dd1-11eb-2275-27380c4af131
# ╠═e15270ba-2dd1-11eb-0c92-a52dd7c668e6
# ╠═e155f280-2dd1-11eb-3e21-4b7b624c5b91
# ╠═e15621ba-2dd1-11eb-2aa3-77152d5a0477
# ╠═e15c4004-2dd1-11eb-0803-81cdc172c069
# ╠═e16045dc-2dd1-11eb-36da-bf2c77219ab8
# ╠═f5bd4480-2dd1-11eb-0535-ffba26a30be2
# ╠═f588c5e8-2dd1-11eb-1859-67bb88f8fbed
