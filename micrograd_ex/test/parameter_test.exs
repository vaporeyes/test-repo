defmodule Micrograd.ParameterTest do
  use ExUnit.Case, async: true
  alias Micrograd.NN.Parameter
  alias Micrograd.{Ops, Topo, Value}

  test "parameter wraps value" do
    p = Parameter.new(1.5)
    assert %Value{} = p.value
    assert Parameter.data(p) == 1.5
  end

  test "grad flows through computation" do
    p1 = Parameter.new(1.0)
    p2 = Parameter.new(2.0)
    loss = Ops.mul(p1.value, p2.value)
    loss = Value.backward(loss)
    leaves = Topo.sort(loss) |> Enum.filter(&(MapSet.size(&1.prev) == 0))
    [v1, v2] = Enum.sort_by(leaves, & &1.data)
    assert v1.grad != 0.0
    assert v2.grad != 0.0
  end
end
