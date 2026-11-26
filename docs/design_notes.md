# Design Notes — Phase 1

## Why MurmurHash3?

We chose MurmurHash3 (x64-128 variant) as our hash function for several
reasons:

1. **HyperLogLog++ precedent**: The HyperLogLog++ paper (Heule et al.,
   Google 2013) uses MurmurHash3 and its analysis relies on the hash
   function's properties.  Using the same hash ensures our implementation
   is directly comparable to published results.

2. **Excellent avalanche properties**: MurmurHash3 is designed so that
   flipping any single input bit flips approximately 50% of the output
   bits.  This is critical for HyperLogLog, which infers cardinality from
   the distribution of leading zeros.

3. **Well-specified reference**: The canonical C implementation by Austin
   Appleby is concise, unambiguous, and widely cross-checked.  The `mmh3`
   Python C extension wraps the same code, giving us a reliable test oracle.

4. **Speed**: While our pure-Python implementation is slower than the C
   version, it correctly implements the algorithm and can be verified.
   In production, one would swap in `mmh3` or a Cython wrapper.

## Why 64-bit hashes?

HyperLogLog needs `p + rank_bits` bits from each hash, where `p ≤ 18` and
`rank_bits ≤ 6` (for 64-bit hashes, the maximum rank is 65).  So 24 bits
suffice for the sketch.

We use 64 bits (the lower half of the 128-bit MurmurHash3 output) because:

- **Collision safety**: For a stream of *n* distinct items, the probability
  of any hash collision in 64 bits is approximately *n²/2⁶⁵*.  At
  *n* = 10⁸, this is ~5 × 10⁻⁴ — negligible for our use case.

- **Future-proofing**: 64 bits gives headroom for larger cardinalities
  or other uses (e.g., Count-Min Sketch row indexing).

- **Standard practice**: Most modern HyperLogLog libraries use 64-bit
  hashes (Redis, Apache DataSketches, Google Zetasketch).

## Why Zipf for traffic simulation?

Real web traffic follows a power-law distribution:

- A small number of pages receive a disproportionate share of views
  (the "long tail").
- Some visitors are heavy users who return frequently.

Zipf's law `P(rank = r) ∝ 1/r^s` is the standard model for this in
information retrieval.  The exponent `s` controls how skewed the
distribution is:

- `s = 0`: uniform (all pages/visitors equally likely).
- `s = 1`: classic Zipf.
- `s > 1`: more skewed (a few items dominate).

We default to `s = 1.1` for pages, which matches empirical observations
of web-page access patterns (Breslau et al., 1999; Adamic & Huberman, 2002).

## `deep_size_bytes` — memory measurement

We use recursive `sys.getsizeof` (with an `id`-based seen set to avoid
double-counting) rather than `tracemalloc` because:

- `sys.getsizeof` is deterministic and reports per-object sizes.
- `tracemalloc` measures process-wide allocations and can include noise
  from the interpreter itself.
- For our purpose — *comparing* memory between exact counters and
  probabilistic sketches — relative accuracy matters more than absolute
  precision.

Limitations: does not account for `__slots__`, C-extension internals,
or numpy array backing stores that bypass `pymalloc`.

## Phase roadmap

| Phase | Description |
|-------|-------------|
| 1 | Foundation: hash, generator, exact counters, interfaces |
| 2 | HyperLogLog implementation and accuracy experiments |
| 3 | Count-Min Sketch implementation |
| 4 | Sketch merging across time windows |
| 5 | FastAPI REST service |
| 6 | Real-time dashboard |
| 7 | Comprehensive benchmarks |
| 8 | Final documentation and presentation |
