defmodule Micrograd.Ops do
  alias Micrograd.Value

  @spec add(Value.t(), Value.t()) :: Value.t()
  def add(x, y) do
    %Value{
      data: x.data + y.data,
      prev: MapSet.new([x, y]),
      op: :add,
      backward: fn self, acc ->
        x_v = Map.get(acc, x, x)
        y_v = Map.get(acc, y, y)

        acc
        |> Map.put(x, %{x_v | grad: x_v.grad + self.grad})
        |> Map.put(y, %{y_v | grad: y_v.grad + self.grad})
      end
    }
  end

  @spec mul(Value.t(), Value.t()) :: Value.t()
  def mul(x, y) do
    %Value{
      data: x.data * y.data,
      prev: MapSet.new([x, y]),
      op: :mul,
      backward: fn self, acc ->
        x_v = Map.get(acc, x, x)
        y_v = Map.get(acc, y, y)

        acc
        |> Map.put(x, %{x_v | grad: x_v.grad + self.grad * y_v.data})
        |> Map.put(y, %{y_v | grad: y_v.grad + self.grad * x_v.data})
      end
    }
  end

  @spec sub(Value.t(), Value.t()) :: Value.t()
  def sub(x, y) do
    %Value{
      data: x.data - y.data,
      prev: MapSet.new([x, y]),
      op: :sub,
      backward: fn self, acc ->
        x_v = Map.get(acc, x, x)
        y_v = Map.get(acc, y, y)

        acc
        |> Map.put(x, %{x_v | grad: x_v.grad + self.grad})
        |> Map.put(y, %{y_v | grad: y_v.grad - self.grad})
      end
    }
  end

  @spec div(Value.t(), Value.t()) :: Value.t()
  def div(x, y) do
    %Value{
      data: x.data / y.data,
      prev: MapSet.new([x, y]),
      op: :div,
      backward: fn self, acc ->
        x_v = Map.get(acc, x, x)
        y_v = Map.get(acc, y, y)

        acc
        |> Map.put(x, %{x_v | grad: x_v.grad + self.grad / y_v.data})
        |> Map.put(y, %{y_v | grad: y_v.grad - self.grad * x_v.data / :math.pow(y_v.data, 2)})
      end
    }
  end

  @spec pow(Value.t(), float) :: Value.t()
  def pow(x, p) do
    data = :math.pow(x.data, p)
    %Value{
      data: data,
      prev: MapSet.new([x]),
      op: :pow,
      backward: fn self, acc ->
        x_v = Map.get(acc, x, x)
        acc
        |> Map.put(x, %{x_v | grad: x_v.grad + self.grad * p * :math.pow(x_v.data, p - 1)})
      end
    }
  end

  @spec neg(Value.t()) :: Value.t()
  def neg(x) do
    %Value{
      data: -x.data,
      prev: MapSet.new([x]),
      op: :neg,
      backward: fn self, acc ->
        x_v = Map.get(acc, x, x)
        Map.put(acc, x, %{x_v | grad: x_v.grad - self.grad})
      end
    }
  end

  @spec exp(Value.t()) :: Value.t()
  def exp(x) do
    data = :math.exp(x.data)
    %Value{
      data: data,
      prev: MapSet.new([x]),
      op: :exp,
      backward: fn self, acc ->
        x_v = Map.get(acc, x, x)
        Map.put(acc, x, %{x_v | grad: x_v.grad + self.grad * data})
      end
    }
  end

  @spec tanh(Value.t()) :: Value.t()
  def tanh(x) do
    data = :math.tanh(x.data)
    %Value{
      data: data,
      prev: MapSet.new([x]),
      op: :tanh,
      backward: fn self, acc ->
        x_v = Map.get(acc, x, x)
        Map.put(acc, x, %{x_v | grad: x_v.grad + self.grad * (1 - data * data)})
      end
    }
  end

  @spec relu(Value.t()) :: Value.t()
  def relu(x) do
    data = if x.data > 0, do: x.data, else: 0.0
    %Value{
      data: data,
      prev: MapSet.new([x]),
      op: :relu,
      backward: fn self, acc ->
        x_v = Map.get(acc, x, x)
        grad = if x_v.data > 0, do: self.grad, else: 0.0
        Map.put(acc, x, %{x_v | grad: x_v.grad + grad})
      end
    }
  end
end
