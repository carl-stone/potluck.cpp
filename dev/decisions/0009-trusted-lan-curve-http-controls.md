# ADR 0009: Trusted-LAN CURVE and HTTP controls

- Date: 2026-08-22
- Status: Accepted
- Decision owner: Carl Stone
- Supersedes: nothing

## Context

Potluck uses a direct adjacent-peer ZeroMQ ring and automatically launches
selected workers through the existing authenticated SSH channel. The initial
deployment boundary is a trusted LAN, but LAN reachability alone must not let an
unauthenticated device join the ring or use a configured HTTP API. The SSH
channel is already the trusted bootstrap path for workers. The security
baseline must preserve the normal household-server flow without requiring user
managed key files or a public-internet security claim.

## Threat boundary

The trusted boundary is the LAN together with the existing SSH host and user
trust. A device admitted by discovery and SSH is trusted to run Potluck code.
This decision does not protect a compromised head, worker, SSH account, or host
operating system, and it does not make a public-network deployment safe. CURVE
protects ring traffic from unauthenticated LAN peers, but it does not make an
admitted host trustworthy. An HTTP API key is a bearer credential and does not
encrypt HTTP traffic.

The HTTP service binds to all interfaces by default (`0.0.0.0`) so household
clients can connect. `--host` can restrict the service to loopback or another
explicit address. Startup must state the bound address and that LAN clients can
connect.

## Decision

### Direct-ring credentials

1. For every topology generation, the head creates an ephemeral in-memory Z85
   ZeroMQ CURVE keypair for every selected ring peer. The credential set is
   never read from a model, configuration, or key file.
2. Each selected peer receives its own public key and secret key, plus the
   expected public key of its next peer, through a one-shot bootstrap record on
   that peer's authenticated SSH stdin before it joins the ring. The local peer
   uses an equivalent in-memory handoff. The record is consumed once and is
   never persisted.
3. A receiving socket is a CURVE server configured with its own keypair. A
   sending socket is a CURVE client configured with its own keypair and the
   expected public key of the receiving peer. Adjacent peers communicate
   directly; the head configures credentials but never relays ring payloads.
4. Secret material must not appear in command-line arguments, environment
   variables, process listings, logs, readiness output, model files, persistent
   worker files, or build artifacts. The head keeps the credential set only in
   memory. A worker keeps only its assigned secret key in process memory and
   erases it on clean shutdown. Logs and errors redact all key material.

### Key lifecycle and fail-closed behavior

A credential generation covers exactly one ring topology. Startup and every
rebuild generate fresh keypairs for all selected peers, close sockets from the
old generation, and reject stale keys. Recovery obtains fresh credentials over
SSH before reconnecting a worker. Rebuilding the topology rotates all keys.

Key-generation failure, an SSH bootstrap failure, missing or malformed
credentials, a peer-authentication failure, or a public-key mismatch excludes
or fails the affected peer and surfaces a security bootstrap error. The ring
must not continue with an unauthenticated peer. There is no unauthenticated
ZeroMQ fallback, alternate transport, local-worker bypass, or secret handoff
through a command line or persistent file. The health state reports rebuilding
or security-bootstrap-failed as appropriate. A non-streaming in-flight request
fails with a retryable error. A streaming request emits a terminal SSE error,
and the client must discard partial output before retrying.

### HTTP API key and CORS

`--api-key` is optional. When configured, every HTTP route requires
`Authorization: Bearer <value>`, including health and model routes; there is no
route exemption. The server compares the value without timing leaks and never
writes it to logs or persistent model or configuration files. When no key is
configured, HTTP remains unauthenticated for the trusted-LAN boundary.

`--cors-origin` accepts one exact allowed origin. When it is omitted, the
server sends no CORS allow-origin header. Wildcard origins are not used, and
credentialed CORS is not enabled. CORS is not authentication.

### HTTP encryption

The trusted-LAN baseline has no built-in TLS. Non-loopback HTTP must not be
treated as safe for the public Internet. A deployment that needs Internet-safe
HTTP must put TLS outside this Potluck baseline.

## Consequences

- Every selected direct-ring peer requires CURVE credentials before it can
  exchange ZeroMQ traffic.
- Topology rebuilds rotate credentials and sockets as one generation; stale
  workers cannot reconnect with old keys.
- SSH trust remains the bootstrap boundary, and secret handling must stay
  confined to the one-shot stdin or local in-memory handoff.
- A configured HTTP API key protects every route, while exact-origin,
  non-credentialed CORS provides browser origin control without being treated
  as authentication.
- Trusted-LAN HTTP remains simple and does not add a built-in TLS stack. Public
  network exposure requires an external TLS boundary and is outside this ADR.
