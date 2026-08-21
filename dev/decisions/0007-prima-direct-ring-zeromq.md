# ADR 0007: Prima direct ring over ZeroMQ

- Date: 2026-08-21
- Status: Accepted
- Decision owner: Carl Stone
- Supersedes: ADR 0005
- Amends: ADR 0006 transport and ring topology

## Context

Prima.cpp is the behavioral progenitor of Potluck's distributed runtime. Its
workers communicate directly in a ring: each selected rank sends to the next
rank, and the final rank returns to rank 0. It uses ZeroMQ as the
message-oriented communication layer for cross-device data and control.

ADR 0005 replaced ZeroMQ with a custom length-prefixed PTLK protocol over raw
TCP. It was written and marked accepted without the decision owner's approval.
It also claimed that every worker connected to its next hop, while the current
`potluck-head --ring` implementation connects the coordinator to every worker
and relays every intermediate hop. ADR 0005 is therefore not an authorized or
accurate product decision.

## Decision

Potluck must use prima.cpp's direct peer-to-peer ring behavior and ZeroMQ
communication model.

1. Each selected device is one ring peer. It connects to its next selected peer
   and receives from its previous selected peer. Rank 0 participates in that
   ring and also owns the client-facing server and control responsibilities.
2. Hidden states and other per-request data travel directly between adjacent
   ring peers. The head must not relay intermediate data for windows it does not
   execute.
3. Cross-device communication uses ZeroMQ message-oriented sockets and follows
   prima.cpp's documented and coded data-plane, control-plane, queueing,
   backpressure, polling, connection, and topology-change behavior.
4. Potluck may define payloads needed by modern llama.cpp and its supported
   models. Exact prima.cpp wire compatibility is not required, but payload or
   implementation differences must not change the required communication
   behavior.
5. The product controller still hides ranks, addresses, ports, and launch order
   from users. Automatic discovery and lifecycle management configure the ring.
6. The custom raw-TCP transport, coordinator-relayed ring route, and PTLK
   length-prefixed data plane are conflicting legacy implementation. They must
   be removed from product source and rewritten tests during the clean cutover.
   They must not remain as fallbacks, diagnostics, compatibility modes, or
   alternate transports.

## Consequences

- ZeroMQ becomes a required Potluck runtime and build dependency on supported
  platforms.
- Ring traffic follows the direct peer path that Potluck inherits from
  prima.cpp, instead of adding a head relay to every window transition.
- Failure handling, worker removal, ring rebuilding, and continuous batching
  must be designed around ZeroMQ ring peers and checked through the integrated
  server.
- Current PTLK/raw-TCP component checks remain historical evidence only. They
  do not satisfy the product contract and must not survive the transport
  cutover.
- Authentication and encryption remain separate security requirements. Using
  ZeroMQ does not by itself make a trusted-LAN deployment safe for the public
  Internet.
