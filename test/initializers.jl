import DiffEqRNN:limit_cycle, orthogonal_init, state0_init

@testset "Type check" begin
    dims = (10,10)
      for init ∈ [limit_cycle, orthogonal_init,state0_init]
        v = init(dims...)
        @test eltype(v) == Float32
      end
    end # type stability

@test_throws AssertionError limit_cycle(11,11)

