defmodule Micrograd.BackwardTest do
  use ExUnit.Case, async: true
  alias Micrograd.{Value, Ops}

  defp f(x, y) do
    Ops.mul(Ops.add(x, y), x)
  end

  defp numeric_grad(x, y, eps) do
    dx =
      (f(Value.new(x + eps), Value.new(y)).data -
         f(Value.new(x - eps), Value.new(y)).data) / (2 * eps)

    dy =
      (f(Value.new(x), Value.new(y + eps)).data -
         f(Value.new(x), Value.new(y - eps)).data) / (2 * eps)

    {dx, dy}
  end

  test "analytic grads match numeric" do
    x = 5.0
    y = 3.0
    root = f(Value.new(x), Value.new(y))
    out = Value.backward(root)
    leaves =
      out
      |> Micrograd.Topo.sort()
      |> Enum.filter(&(MapSet.size(&1.prev) == 0))
      |> Enum.sort_by(& &1.data)
    [leaf_x, leaf_y] = leaves
    {dx, dy} = numeric_grad(x, y, 1.0e-6)
    assert_in_delta leaf_x.grad, dx, 1.0e-4
    assert_in_delta leaf_y.grad, dy, 1.0e-4
  end

  test "double backward accumulates" do
    x = 5.0
    y = 3.0
    root = f(Value.new(x), Value.new(y))
    out1 = Value.backward(root)
    out2 = Value.backward(out1)
    leaves =
      out2
      |> Micrograd.Topo.sort()
      |> Enum.filter(&(MapSet.size(&1.prev) == 0))
      |> Enum.sort_by(& &1.data)
    [leaf_x, leaf_y] = leaves
    {dx, dy} = numeric_grad(x, y, 1.0e-6)
    assert_in_delta leaf_x.grad, 2 * dx, 1.0e-4
    assert_in_delta leaf_y.grad, 2 * dy, 1.0e-4
  end
end
