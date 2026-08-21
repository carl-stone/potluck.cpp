# ADR 0005: Raw TCP, not ZeroMQ

- Date: 2026-08-20
- Status: Superseded by ADR 0007
- Approval note: This decision was marked accepted without the decision owner's
  approval and was never a binding Potluck product decision.

## Historical context

The unfinished component implementation needed ordered message delivery on a
trusted LAN. An agent selected raw POSIX TCP instead of prima.cpp's ZeroMQ
communication model.

## Superseded decision

The implementation used a custom length-prefixed PTLK frame protocol over raw
TCP. The ADR also claimed that each worker connected to one next hop, but the
implemented `potluck-head --ring` path instead connected the coordinator to
every worker and relayed every hop.

ADR 0007 requires prima.cpp's direct peer-to-peer ring behavior and ZeroMQ
communication model. The custom raw-TCP transport and coordinator-relayed route
are removal targets, not supported alternatives.

## Historical consequences

- The component path avoided a ZeroMQ dependency.
- It had to implement framing, partial reads and writes, timeouts, and retries.
- It did not provide authentication or encryption.
- Its transport and topology checks are historical evidence only.
