defmodule Micrograd.OpsBasicTest do
  use ExUnit.Case, async: true
  alias Micrograd.{Value, Ops}

  test "basic ops forward" do
    x = Value.new(3)
    y = Value.new(2)
    assert Ops.add(x, y).data == 5.0
    assert Ops.mul(x, y).data == 6.0
  end

  test "numeric gradient check" do
    f = fn xv, yv ->
      Ops.mul(Ops.add(xv, yv), xv)
    end

    x = Value.new(3.0)
    y = Value.new(2.0)
    out = f.(x, y)
    out = Value.backward(out)
    [x2, y2] = Enum.sort_by(out.prev, & &1.data)

    eps = 1.0e-6
    num_dx =
      (f.(Value.new(x.data + eps), Value.new(y.data)).data -
         f.(Value.new(x.data - eps), Value.new(y.data)).data) / (2 * eps)

    num_dy =
      (f.(Value.new(x.data), Value.new(y.data + eps)).data -
         f.(Value.new(x.data), Value.new(y.data - eps)).data) / (2 * eps)

    assert_in_delta x2.grad, num_dx, 1.0e-4
    assert_in_delta y2.grad, num_dy, 1.0e-4
  end
end
