# Count-Min Sketch Theory & Worked Example

## 1. Intuition

A **Count-Min Sketch (CMS)** estimates the frequency of items in a stream
using a fixed-size $d \times w$ matrix of counters, where $d$ is the **depth**
(number of independent hash functions) and $w$ is the **width** (number of
buckets per row).

Every incoming item $x$ hashes to one column $h_i(x) \in \{0, \dots, w-1\}$
in each row $i \in \{0, \dots, d-1\}$, and all $d$ counters are incremented.

---

## 2. Why Take the Minimum?

Because multiple items can collide in the same bucket, every counter
$\text{table}[i, h_i(x)]$ is an **upper bound** on the true frequency $f(x)$:

$$\text{table}[i, h_i(x)] = f(x) + \text{noise}_i \ge f(x)$$

Since noise is always non-negative (for non-negative updates), the **minimum**
across all $d$ rows gives the tightest upper bound:

$$\hat{f}(x) = \min_{0 \le i < d} \text{table}[i, h_i(x)] \ge f(x)$$

---

## 3. $(\varepsilon, \delta)$ Guarantee — Proof Sketch

Choose $w = \lceil e / \varepsilon \rceil$ and $d = \lceil \ln(1 / \delta) \rceil$.
Let $N = \sum_y f(y)$ be the total count in the stream.

1. **Expected collision noise in one row**: Since $h_i$ is uniform over $w$
   buckets, the expected noise added to $x$'s bucket in row $i$ is:
   $$\mathbb{E}[\text{noise}_i] = \frac{N - f(x)}{w} \le \frac{N}{w}$$
2. **Markov's inequality** on non-negative $\text{noise}_i$:
   $$\Pr\bigl[\text{noise}_i > \varepsilon N\bigr] \le \frac{\mathbb{E}[\text{noise}_i]}{\varepsilon N} \le \frac{N / w}{\varepsilon N} = \frac{1}{\varepsilon w} \le \frac{1}{e}$$
3. **Independence across $d$ rows**: The minimum exceeds $f(x) + \varepsilon N$
   only if **all $d$ rows** simultaneously exceed $f(x) + \varepsilon N$:
   $$\Pr\bigl[\hat{f}(x) - f(x) > \varepsilon N\bigr] \le \left(\frac{1}{e}\right)^d = e^{-d} \le \delta$$

---

## 4. Conservative Update

Standard CMS increments all $d$ counters by `count`. **Conservative Update**
(Estan & Varghese, 2002) reduces over-estimation by first computing the
current estimate $e = \hat{f}(x)$ and then setting:

$$\text{table}[i, h_i(x)] \leftarrow \max\bigl(\text{table}[i, h_i(x)],\; e + \text{count}\bigr)$$

Counters that are already inflated by collisions are not incremented further,
keeping $\hat{f}(x) \ge f(x)$ while substantially tightening estimates on
skewed (Zipf) streams. Note that conservative updates are **order-dependent**.

---

## 5. Worked $3 \times 5$ Example

Suppose $d = 3$, $w = 5$, all counters start at `0`, and we process:
`("a", 2)`, `("b", 3)`, `("c", 1)` with hashes:
- `a`: cols `[1, 3, 0]`
- `b`: cols `[1, 2, 4]` (collides with `a` in row 0!)
- `c`: cols `[2, 3, 1]` (collides with `a` in row 1!)

| Step | Standard CMS table (rows 0..2) | Conservative CMS table (rows 0..2) |
|---|---|---|
| After `("a", 2)` | `r0:[0,2,0,0,0]` `r1:[0,0,0,2,0]` `r2:[2,0,0,0,0]` | identical (`e=0`, target `2`) |
| After `("b", 3)` | `r0:[0,5,0,0,0]` `r1:[0,0,3,2,0]` `r2:[2,0,0,0,3]` | `e=min(2,0,0)=0`, target `3` → `r0:[0,3,0,0,0]` `r1:[0,0,3,2,0]` `r2:[2,0,0,0,3]` |
| After `("c", 1)` | `r0:[0,5,1,0,0]` `r1:[0,0,3,3,0]` `r2:[2,1,0,0,3]` | `e=min(0,2,0)=0`, target `1` → `r0:[0,3,1,0,0]` `r1:[0,0,3,2,0]` `r2:[2,1,0,0,3]` |

Querying `"a"` (`cols=[1, 3, 0]`):
- Standard: `min(5, 3, 2) = 2` (row 0 was inflated to `5`).
- Conservative: `min(3, 2, 2) = 2` (row 0 only rose to `3`, reducing noise for future items).
