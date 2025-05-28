defmodule Micrograd.TopoTest do
  use ExUnit.Case, async: true
  alias Micrograd.{Value, Ops, Topo}

  test "topological sort order" do
    x = Value.new(3.0)
    y = Value.new(2.0)
    add = Ops.add(x, y)
    root = Ops.mul(add, x)

    sorted = Topo.sort(root)
    assert List.last(sorted) == root

    positions = Enum.with_index(sorted) |> Map.new(fn {n, i} -> {n, i} end)

    Enum.each(sorted, fn node ->
      Enum.each(node.prev, fn parent ->
        assert positions[parent] < positions[node]
      end)
    end)
  end
end
