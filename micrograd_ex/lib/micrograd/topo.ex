defmodule Micrograd.Topo do
  alias Micrograd.Value

  @spec sort(Value.t()) :: [Value.t()]
  def sort(root) do
    {order, _} = dfs(root, MapSet.new(), [])
    Enum.reverse(order)
  end

  defp dfs(%Value{} = v, visited, acc) do
    if MapSet.member?(visited, v) do
      {acc, visited}
    else
      visited = MapSet.put(visited, v)
      {acc, visited} =
        Enum.reduce(v.prev, {acc, visited}, fn child, {acc, vis} ->
          dfs(child, vis, acc)
        end)
      {[v | acc], visited}
    end
  end
end
