@testset "Checking type stability" begin
    hsize = 10
    bsize = 3
    x = randn(Float32, hsize,bsize)
    for ∂rnn ∈ [∂RNNCell,∂GRUCell,∂LSTMCell]
        rnn = ∂rnn(hsize,hsize)
        if typeof(rnn) <: ∂LSTMCell
            h = randn(Float32, 2hsize,bsize)
        else
            h = randn(Float32, hsize,bsize)
        end
        @test eltype(rnn(h,x,rnn.initial_params())) == Float32
    end
end