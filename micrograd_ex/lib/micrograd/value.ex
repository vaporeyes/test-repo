defmodule Micrograd.Value do
  @moduledoc """
  Scalar value in the autograd graph.
  """

  defstruct data: 0.0, grad: 0.0, prev: MapSet.new(), op: :nil, backward: fn -> :ok end

  @type t :: %__MODULE__{
          data: float,
          grad: float,
          prev: MapSet.t(t),
          op: atom,
          backward: (() -> any())
        }

  @spec new(float) :: t
  def new(x) when is_number(x) do
    %__MODULE__{data: x * 1.0}
  end

  @spec zero_grad(t) :: t
  def zero_grad(%__MODULE__{prev: prev} = v) do
    new_prev =
      prev
      |> Enum.map(&zero_grad/1)
      |> MapSet.new()

    %{v | grad: 0.0, prev: new_prev}
  end

  defimpl String.Chars do
    def to_string(%{data: d}), do: "#{d}"
  end

  @spec backward(t) :: t
  def backward(root) do
    root = zero_grad(root)
    topo = Micrograd.Topo.sort(root)
    map = Map.new(topo, &{&1, &1})
    map = Map.update!(map, root, &%{&1 | grad: 1.0})

    map =
      Enum.reduce(Enum.reverse(topo), map, fn node, acc ->
        node_copy = Map.get(acc, node)
        node_copy.backward.(node_copy, acc)
      end)

    {map, _} =
      Enum.reduce(topo, {map, nil}, fn node, {m, _} ->
        copy = Map.get(m, node)
        new_prev =
          copy.prev
          |> Enum.map(&Map.get(m, &1))
          |> MapSet.new()
        updated = %{copy | prev: new_prev}
        {Map.put(m, node, updated), nil}
      end)

    Map.get(map, root)
  end
end
