defmodule Micrograd.ValueTest do
  use ExUnit.Case, async: true

  alias Micrograd.Value

  test "new/1 defaults" do
    v = Value.new(3.0)
    assert v.data == 3.0
    assert v.grad == 0.0
    assert v.prev == MapSet.new()
  end

  test "zero_grad/1 resets nested grads" do
    x = %{Value.new(1.0) | grad: 0.5}
    y = %{Value.new(2.0) | grad: 0.25}
    root = %{Value.new(3.0) | grad: 1.0, prev: MapSet.new([x, y])}

    reset = Value.zero_grad(root)

    [a, b] = Enum.sort_by(reset.prev, & &1.data)
    assert reset.grad == 0.0
    assert a.grad == 0.0
    assert b.grad == 0.0
  end
end
