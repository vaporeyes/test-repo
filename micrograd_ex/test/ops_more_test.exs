defmodule Micrograd.OpsMoreTest do
  use ExUnit.Case, async: true
  alias Micrograd.{Value, Ops}

  @x 1.23

  test "forward ops" do
    x = Value.new(@x)
    y = Value.new(2.0)
    assert Ops.sub(x, y).data == @x - 2.0
    assert Ops.div(x, y).data == @x / 2.0
    assert Ops.pow(x, 3.0).data == :math.pow(@x, 3.0)
    assert Ops.neg(x).data == -@x
    assert Ops.exp(x).data == :math.exp(@x)
    assert Ops.tanh(x).data == :math.tanh(@x)
    relu_expected = if @x > 0, do: @x, else: 0.0
    assert Ops.relu(x).data == relu_expected
  end

  test "gradients for composed function" do
    f = fn xv ->
      Ops.tanh(Ops.exp(Ops.neg(Ops.pow(xv, 2.0))))
    end

    x0 = Value.new(@x)
    out = f.(x0)
    out = Value.backward(out)
    [leaf] = Micrograd.Topo.sort(out) |> Enum.filter(&(MapSet.size(&1.prev) == 0))

    eps = 1.0e-6
    num_dx =
      (f.(Value.new(@x + eps)).data - f.(Value.new(@x - eps)).data) / (2 * eps)

    assert_in_delta leaf.grad, num_dx, 1.0e-4
  end
end
