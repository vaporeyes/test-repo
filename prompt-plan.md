## Prompt Plan for `micrograd_ex`

### C0-S1 · Mix Scaffold & CI

```text
You are implementing the Elixir library **micrograd_ex** (reverse-mode AD for scalars).

## Task
1. Create a new Mix project:  
   ```bash
   mix new micrograd_ex --sup
```

2. Configure `.formatter.exs` and preferred `mix test` flags (`--warnings-as-errors`).

3. Add `ci.yml` GitHub Action (OTP 26 & 27).

## Acceptance

✅ `mix compile --warnings-as-errors` succeeds.  
✅ `mix test` runs with 0 tests, 0 failures.  
✅ `.github/workflows/ci.yml` is present, uses `actions/setup-elixir`.  
Return only the code you added or changed.

````

---

# Chunk C1 – Core Scalar Type

### C1-S1 · `Micrograd.Value` struct & helpers  

```text
You will create the heart of the AD engine: the immutable `Micrograd.Value` struct.

## Requirements
* Fields: `data :: float`, `grad :: float`, `prev :: MapSet.t(Value.t)`, `op :: atom`, `backward :: (() -> any())`.
* Public API:
  * `new/1`
  * `zero_grad/1` – deep-reset grads to 0.0
  * `impl String.Chars` to output `"#{data}"` for iex convenience

## Tests
Write `test/value_test.exs`:
1. `new/1` stores data & defaults (`grad == 0.0`, empty `prev`).
2. `zero_grad/1` sets nested grads to 0.0 (build a toy graph manually).

## Acceptance
✅ All tests green.  
✅ Dialyzer passes (`mix dialyzer`) though dialyzer config can be minimal for now.  
Return only new/modified files.
````

---

# Chunk C2 – Basic Arithmetic Ops

### C2-S1 · `Micrograd.Ops.add/2` & `mul/2`

````text
Implement the first two differentiable operations.

## API
```elixir
alias Micrograd.{Value, Ops}
Ops.add(x :: Value.t(), y :: Value.t()) :: Value.t()
Ops.mul(x :: Value.t(), y :: Value.t()) :: Value.t()
````

## Behaviour

- Compose a new `Value` with correct `data`, `prev`, `op`.

- `backward` closure updates parent grads (immutably: return new parent copies).

## Tests (`test/ops_basic_test.exs`)

1. Forward checks: 3 + 2 == 5, 3 * 2 == 6.

2. Numerical gradient check

    - `f(x, y) = (x + y) * x` at (x=3, y=2)

    - Compare analytical grads after `Value.backward/1` against central-difference numeric grad (ε=1e-6, tolerance 1e-4).

## Acceptance

✅ Tests pass; numeric vs analytic difference < 1e-4.  
✅ No other ops yet.  
Return only changed files.

````

---

# Chunk C3 – Topological Sort & Global Back-prop

### C3-S1 · `Micrograd.Topo.sort/1`

```text
Implement a DFS post-order topological sort for an arbitrary `Value` graph.

## Contract
`Topo.sort(root :: Value.t()) :: [Value.t()]` parents before children.

## Tests
1. Feed the graph from C2 numeric-gradient test; assert last element is `root`.
2. Assert every node’s parents appear *earlier* in the list.

### C3-S2 · `Value.backward/1`

Extend `Micrograd.Value`:

* `backward/1` zeroes grads, sets `root.grad = 1.0`, iterates reverse topo list, executing each node’s stored `backward` closure.
* Use an object-id-to-copy map so parent copies with updated grads propagate.

## Tests (`test/backward_test.exs`)
1. Re-use function f from earlier; assert gradients match numeric check (x → 5, y → 3).
2. Double-call safety: calling `Value.backward/1` twice without `zero_grad/1` should accumulate grads (documented behaviour).

## Acceptance
✅ All topo & backward tests green.  
Return diffs only.
````

---

# Chunk C4 – More Ops & Activations

### C4-S1 · Remaining scalar ops

```text
Add Ops: `sub/2`, `div/2`, `pow/2`, `neg/1`, `exp/1`, `tanh/1`, `relu/1`.

## Tests
1. Forward values for each op against :math.*
2. Analytic vs numeric gradient for `f(x)=tanh(exp(-x^2))` at x = 1.23

## Acceptance
✅ Grad diff < 1e-4.  
Return only new functions/tests.
```

---

# Chunk C5 – Parameter Wrapper

### C5-S1 · `Micrograd.NN.Parameter`

```text
Define:
defstruct [:value]     # Value.t()

API
* `new/1` – wraps float into Value
* `data/1`, `grad/1`

## Tests
* Creating a Parameter stores the underlying Value.
* Grad flows: build two parameters, compute simple loss, backprop, ensure param.grad ≠ 0.
```

---

# Chunk C6 – Neuron & Layer

### C6-S1 · `Micrograd.NN.Neuron`

```text
Neuron.new(in_dim, activation \\ :tanh)
* Random weights U(-1,1) + bias

forward/2 – returns single Value (activation(Σ w·x + b))

Tests:
1. Deterministic with `:rand.seed/2`.
2. Shape: forward outputs Value, params count == in_dim + 1.
```

### C6-S2 · `Micrograd.NN.Layer`

```text
Layer.new(in_dim, out_dim, act)
Layer.forward(layer, xs :: [Value.t()]) :: [Value.t()]

Tests:
* For input length == in_dim, output length == out_dim.
* Gradients propagate through a 2-layer stack on XOR sample.
```

---

# Chunk C7 – MLP & End-to-End Loss

### C7-S1 · `Micrograd.NN.MLP`

```text
MLP.new(in_dim, hidden_sizes :: [pos_integer], act \\ :tanh)
forward/2 returns list of outputs (one per last layer neuron)

parameters/1 flattens nested Parameter lists.

Tests:
1. Forward pass on zeroed weights returns bias value only.
2. Backprop on tiny dataset reduces loss after one SGD step.
```

---

# Chunk C8 – Optimiser (SGD)

### C8-S1 · `Micrograd.Optim.SGD`

```text
step!(params, lr) – returns updated params with weight = weight - lr * grad

Tests:
* With lr=0.1, param (data=1.0, grad=0.5) updates to 0.95.
* Integration: Train XOR for 1 000 epochs; final loss < 0.1.
(To keep test time <2 s, shrink epochs OR seed weights favourable.)
```

---

# Chunk C9 – Public Umbrella API

### C9-S1 · `MicrogradEx` facade

```text
Expose:
* new_value/1
* grad(fn -> loss end, params)
* fit(model, data, epochs, lr)

Tests:
* `grad/2` returns {loss, params_with_grad}
* `fit/4` on line-fit dataset drives MSE below 1e-2.
```

---

# Chunk C10 – Documentation & Coverage

### C10-S1 · ExDoc & doctests

```text
Add `ex_doc` to deps; ensure `mix docs` succeeds.

Write doctests for at least Value, Ops, and MLP.

Add `mix test --cover` to CI; enforce ≥ 95 % coverage.

No extra tests required—CI green is acceptance.
```
