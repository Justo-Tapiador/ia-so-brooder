# Security Policy

## Reporting a Vulnerability

Please report vulnerabilities privately to
[justo.tapiador@gmail.com](mailto:justo.tapiador@gmail.com) (GPG key on
request). Include reproduction steps and the affected commit. We will
respond within 72 hours and credit reporters in the release notes.

Do **not** open public issues for security problems.

## Threat model

IA-SO Brooder deliberately separates two worlds:

* **Trusted computing base (kernel)**: `brooder/nucleo.py`,
  `brooder/primitivas/` — the code that validates and executes every
  primitive against the machine.
* **Untrusted payload (the brain)**: the neural network and its
  weights. The AI never touches hardware directly; it can only
  *request* primitives through the type contract.

The design assumption is that the **brain is data, not code**. These
mechanisms enforce it:

### 1. SSD images are loaded weights-only

`ssd/brooder.img` is an artifact that travels across networks and
machines. Since a `.pt` file is a Python `pickle`, a malicious image
could otherwise execute arbitrary code at boot time on the "birth PC"
(`torch.load` with `weights_only=False` deserializes and runs any
embedded `__reduce__` payload).

Mitigation (enforced and tested in `tests/test_seguridad.py`):

* Every `torch.load` in the codebase uses `weights_only=True`
  (`brooder/cerebro.py`, `brooder/nucleo.py`, `brooder/incubadora.py`).
  Only tensors and primitive Python types are accepted.
* `exportar_ssd` packages **only** `config` + `estado` (weights) into
  the image. Optimizer state and anything else never travel.
* A tampered image can at most fail to load — it cannot execute.

**Trust boundary**: whoever mounts an SSD image still trusts its
*weights* (a hostile producer could ship a useless or garbage brain).
Integrity of the shipped image can be verified with the SHA-256 digest
published in `README.md` and in each release.

### 2. No escape hatch from the primitive contract

* The brain perceives the machine only through `InstanteMaquina`: no
  file paths, no pointers, no callable objects.
* `PCReal` writes are confined to `<sandbox>/disco/0.tok .. 9.tok`;
  slot numbers are validated integers — there is no path constructor.
* There is no primitive for code execution, process spawning, or
  arbitrary filesystem access. The network primitive returns a
  controlled error (disabled by policy).

### 3. Resource limits

Each user request has a fixed cycle budget enforced by the kernel, so
a malfunctioning brain cannot spin the machine forever.

## Hardening checklist for forks

If you extend Brooder with new primitives, keep the invariant:

1. New primitives must validate their arguments **before** touching the
   host (see `MaquinaBase.ejecutar`).
2. Never pass deserialized objects to `eval`/`exec`/`subprocess`;
   never resolve file paths from network state.
3. Any new `torch.load` must use `weights_only=True`.
4. Keep the payload surface minimal when exporting SSD images.
