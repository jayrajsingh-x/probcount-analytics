# HyperLogLog Theory & Worked Example

## 1. Intuition

Suppose you hash every incoming visitor ID uniformly into a 64-bit binary
string.  If the hashes are truly random, then:

- About **1 in 2** hashes start with `1…` (0 leading zeros, $\rho = 1$).
- About **1 in 4** hashes start with `01…` (1 leading zero, $\rho = 2$).
- About **1 in $2^k$** hashes start with $k-1$ zeros followed by `1` ($\rho = k$).

If the maximum $\rho$ we observe across a stream is $R$, we might guess there
were roughly $2^R$ distinct items.  Using a single maximum has huge variance,
so **HyperLogLog** splits the stream into $m = 2^p$ independent buckets
("stochastic averaging") and combines their maxima with a **harmonic mean**.

---

## 2. Register Structure & $\rho$

Given a 64-bit hash $h = \text{hash64}(x)$:

1. **Bucket index**: the top $p$ bits select register $j \in \{0, \dots, m - 1\}$.
2. **Rank source**: the remaining $64 - p$ bits form $w$.
3. **Rank $\rho(w)$**: position of the leftmost `1`-bit in the $(64 - p)$-bit
   field (1-indexed), i.e. `(leading zeros of w) + 1`.
4. **Update**: $M[j] \leftarrow \max(M[j], \rho(w))$.

Because $64 - p \le 60$, $\rho \le 61 < 64$, so every register fits in **6 bits**
(at $p = 14$, $m = 16\,384$ registers take $\lceil 16\,384 \times 6 / 8 \rceil = 12\,288$ bytes = **12 KB**).

---

## 3. Harmonic Mean & Bias Corrections

Let $Z = \left(\sum_{j=0}^{m-1} 2^{-M[j]}\right)^{-1}$.  The raw estimate is:

$$E = \alpha_m \cdot m^2 \cdot Z$$

where $\alpha_{16} = 0.673$, $\alpha_{32} = 0.697$, $\alpha_{64} = 0.709$, and
$\alpha_m = \frac{0.7213}{1 + 1.079/m}$ for $m \ge 128$.

- **Small-range correction (Linear Counting)**: when $E \le 2.5m$ and the
  number of empty registers $V > 0$, we return $m \ln(m / V)$ because the
  harmonic mean overestimates when many registers are still zero.
- **Large-range correction**: unnecessary with a 64-bit hash, as hash
  collisions only saturate near $2^{64}$.
- **Standard error**: $\sigma \approx \frac{1.04}{\sqrt{m}}$ (for $p=14$,
  $\sigma \approx 0.81\%$).

---

## 4. Merging

Two sketches with the same $(p, \text{seed})$ merge by taking the
element-wise maximum of their registers:

$$M_{\text{merged}}[j] = \max\bigl(M_A[j], M_B[j]\bigr)$$

Because $\max$ is commutative, associative, and idempotent, distributed
servers can count local streams independently and merge sketches without
double-counting overlapping visitors.

---

## 5. Worked Hand Example ($p = 4$, $m = 16$ registers)

With $p = 4$, $m = 16$, $\alpha_{16} = 0.673$, and $64 - p = 60$ rank bits.
Suppose we insert 10 items whose 64-bit hashes have the following top-4 bits
(`index`) and leading zeros in their lower 60 bits:

| Step | Item | Top 4 bits (`index`) | Leading 0s in $w$ | $\rho$ | Register $M[\text{index}]$ after |
|---|---|---|---|---|---|
| 1 | `u_01` | `0011` (3) | 0 (`1...`) | 1 | $M[3] = 1$ |
| 2 | `u_02` | `1010` (10) | 2 (`001...`) | 3 | $M[10] = 3$ |
| 3 | `u_03` | `0011` (3) | 1 (`01...`) | 2 | $M[3] = \max(1, 2) = 2$ |
| 4 | `u_04` | `0000` (0) | 0 (`1...`) | 1 | $M[0] = 1$ |
| 5 | `u_05` | `1111` (15) | 3 (`0001...`) | 4 | $M[15] = 4$ |
| 6 | `u_06` | `0110` (6) | 0 (`1...`) | 1 | $M[6] = 1$ |
| 7 | `u_07` | `1010` (10) | 0 (`1...`) | 1 | $M[10] = \max(3, 1) = 3$ |
| 8 | `u_08` | `1001` (9) | 1 (`01...`) | 2 | $M[9] = 2$ |
| 9 | `u_09` | `0011` (3) | 0 (`1...`) | 1 | $M[3] = \max(2, 1) = 2$ (duplicate/smaller ignored) |
| 10 | `u_10` | `1100` (12) | 0 (`1...`) | 1 | $M[12] = 1$ |

Final registers $M[0..15]$: `[1, 0, 0, 2, 0, 0, 1, 0, 0, 2, 3, 0, 1, 0, 0, 4]`
- Non-zero registers: $8$ out of $16$ $\implies V = 8$ zero registers.
- Sum $\sum 2^{-M[j]} = 8(1) + 4(2^{-1}) + 2(2^{-2}) + 1(2^{-3}) + 1(2^{-4}) = 8 + 2 + 0.5 + 0.125 + 0.0625 = 10.6875$.
- Raw estimate: $E = 0.673 \times 16^2 / 10.6875 \approx 16.12 \le 2.5 \times 16 = 40$.
- Since $E \le 40$ and $V = 8 > 0$, Linear Counting applies:
  $$\hat{n} = 16 \ln(16 / 8) = 16 \ln 2 \approx 11.09 \quad (\text{true } n = 10)$$
