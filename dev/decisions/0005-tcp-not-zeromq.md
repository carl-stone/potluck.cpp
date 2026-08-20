# ADR 0005: Raw TCP, not ZeroMQ

- Date: 2026-08-20
- Status: Accepted

## Context

The cluster protocol needs ordered point-to-point delivery of frames between
neighboring workers in a ring, on a trusted LAN. Options were raw POSIX TCP or
a messaging library such as ZeroMQ.

## Decision

Raw TCP with a length-prefixed frame protocol (`potluck-protocol.h`). No
middleware dependency; every worker connects to exactly one next hop.

## Consequences

- Zero third-party runtime dependencies in the data path; builds on macOS and
  Linux with only libc.
- The protocol must handle partial reads/writes itself, which it does
  (`recv_full`/`send_all`).
- No auth or encryption: LAN-trusted deployment only, stated as a limitation.
