"""
Single source of truth for M1's three locked notation symbols.

Per Gate M1-N (m1-notation-gate.md, amended 2026-09-16 Sec 3A) and CLAUDE.md
Sec 1.8, `k`, `N`, and `lambda` (security level; spelled `LAM_` below since
`lambda` is a Python keyword) are locked, repo-wide, forever:

    k      = the degree of the polynomial approximating the non-linear split
             function inside CKKS-encrypted tree evaluation.
    N      = the CKKS polynomial modulus degree (ring dimension), which fixes
             the number of plaintext slots and, jointly with the coefficient
             modulus, the security level.
    lambda = the CKKS security level in bits. A property of N and the
             coefficient-modulus chain, never varied across the k-sweep.

Every other module in this repo that needs to reference any of these symbols
should import the bindings below rather than re-declaring bare `k`/`N`/`lam`
variables, so that `check_notation.py` has one canonical definition to treat
as sanctioned.

N, the coefficient-modulus chain, and lambda are held FIXED across every
k-sweep experiment (they are a property of the encryption context, not of the
approximation). Changing them changes the security level, not the
utility-latency trade-off being characterized -- do not vary them per-k.
"""

# The four approximation polynomial degrees swept in M1's primary research
# contribution (Roadmap PDF Sec M1, required ablations).
K_APPROX_DEGREE_VALUES: tuple[int, ...] = (3, 5, 7, 9)

# The fixed-k operating point used for M1's mandatory Track A core pipeline
# (plaintext -> CKKS encrypted eval), before the k-sweep ablation.
K_MANDATORY_CORE_DEGREE: int = 5

# CKKS polynomial modulus degree (ring dimension). Fixed at Gate M1-0
# sign-off and never varied across k experiments. Derived in
# results/m1_circuit_params.json (Gate M1-0 drafting pass, m1-0-
# implementation-lock.md Sec 5, 2026-09-16) and copied here 2026-09-18 upon
# sign-off (m1-0-implementation-lock.md Sec 13/14 -- relayed by the project
# owner in chat, not a forwarded document; recorded as exactly that).
N_POLY_MODULUS_DEGREE: int | None = 16384

# CKKS security level in bits. Derived in m1-0-implementation-lock.md Sec 5
# and CONFIRMED (not just asserted) against the Homomorphic Encryption
# Security Standard tables via src/verify_ckks_security_level.py -- N=16384's
# 400-bit chain passes SEAL's TC128 enforcement and fails TC192. Copied here
# 2026-09-18 upon the same sign-off as N_POLY_MODULUS_DEGREE above.
LAM_SECURITY_LEVEL: int | None = 128

# Primary Track A utility criterion (Roadmap PDF Sec M1 / CLAUDE.md Sec M1):
# |F1_plaintext - F1_encrypted| must be strictly less than this value.
MAX_ACCEPTABLE_F1_LOSS: float = 0.03
