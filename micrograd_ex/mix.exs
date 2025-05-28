defmodule MicrogradEx.MixProject do
  use Mix.Project

  def project do
    [
      app: :micrograd_ex,
      version: "0.1.0",
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      aliases: aliases(),
      deps: deps()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {MicrogradEx.Application, []}
    ]
  end

  defp deps do
    [
      {:dialyxir, "~> 1.3", only: [:dev], runtime: false}
    ]
  end

  defp aliases do
    [test: ["test --warnings-as-errors"]]
  end
end
