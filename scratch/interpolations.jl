using Random, DiffEqRNN, DataInterpolations
# [CubicSplineRegularGrid, LinearInterpolationRegularGrid,
#    ConstantInterpolationRegularGrid]
A = randn(170,784)
itp = ConstantInterpolationRegularGrid(A)
dump(itp)
# CubicSplineRegularGrid
# itp.z and itp.u has dimensions batch × time
# same for the other RegularGrid interpolations
##
itp = CubicSpline(A, collect(0:783))
dump(itp)
# CubicSpline with 2-D Array input and vector time has the same batch × time dimensions
##
itp = LinearInterpolation(A, collect(0:783))
dump(itp)
# ditto for LinearInterpolation
##
t₁ = 10
bs = 7
inputsize = 3
x = Float32(sqrt(1/2))randn(Float32, inputsize, bs, t₁)
times = cumsum(randexp(Float32, 1, bs, t₁), dims=3)
x = cat(times,x,dims=1)
x = reshape(x, :,t₁)
times = reshape(times, :, t₁)
x = [x[i,:] for i ∈ 1:size(x,1)]
times = repeat(times, inner=(inputsize+1,1))
times = [times[i,:] for i ∈ 1:size(x,1)]
X = CubicSpline(x, times)
dump(X)
