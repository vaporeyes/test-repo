defmodule Micrograd.NN.Parameter do
  @moduledoc false
  alias Micrograd.Value

  defstruct [:value]

  @type t :: %__MODULE__{value: Value.t()}

  @spec new(float) :: t
  def new(x) when is_number(x) do
    %__MODULE__{value: Value.new(x)}
  end

  @spec data(t) :: float
  def data(%__MODULE__{value: %Value{data: d}}), do: d

  @spec grad(t) :: float
  def grad(%__MODULE__{value: %Value{grad: g}}), do: g
end
