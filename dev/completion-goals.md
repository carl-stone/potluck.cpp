# Potluck completion goals

Carl approved these three product goals on 2026-08-24. They define behavior
that must exist before Potluck is ready. They do not choose an implementation.

## Goal 1: Separate conversations

Two chats must be able to run at the same time. Each later message must see
only the earlier messages from its own chat.

### Current state

- Each HTTP request gets a new slot.
- The client must send the full `messages` history each time.
- The server does not keep a conversation ID or reuse a slot for a later
  message.

### Target behavior

- Give each chat a conversation ID.
- Keep each ID linked to its own slot and history.
- Keep two chats independent during concurrent use.
- Restore the needed history after a ring rebuild.
- Define what happens when all slots are in use or a chat is idle.

### Proof

- Run two chats at the same time.
- Send follow up messages to both.
- Confirm that neither chat receives messages from the other.
- Repeat after a ring rebuild.

This is medium to large work. The difficult part is keeping the right state
when the ring changes.

## Goal 2: API cancellation

An ordinary client must be able to stop a request through the normal HTTP
interface. This includes a user who stops a streamed answer by closing the
connection.

### Current state

- The streamed response detects a closed client connection.
- The scheduler marks the request cancelled and stops ring work.
- There is no end to end proof from an ordinary client.
- Cancellation for a request with no stream needs a clear contract.

### Target behavior

- A client can stop a streamed request.
- The server stops the work and frees the slot.
- A new request can use that slot without a hang or mixed state.
- The response and error behavior are documented.
- Decide if an explicit cancel request is needed.

### Proof

- Start a streamed request with an ordinary client.
- Close the connection before completion.
- Check that the slot becomes free.
- Send another request and confirm it completes normally.

Disconnect based cancellation is small to medium work. An explicit cancel
request would be larger.

## Goal 3: Worker refresh recovery

When a remote machine has an old or missing Potluck worker, the server must
update it and restore service without a hang.

### Current state

- The server checks the remote worker build.
- It uses SSH and rsync to update the worker.
- Ring rebuilds retry after failures.
- The current order can stop the old worker before the update succeeds.

### Target behavior

- Copy the new worker to a temporary location.
- Verify the copy before stopping the current worker.
- Keep or relaunch the old worker if the new start fails.
- Leave an unavailable machine out of the next valid ring when possible.
- Let a recovered machine join later.
- Preserve active chat state across the ring change.

### Proof

- Make one remote worker use an old build.
- Cause one update attempt to fail.
- Restore the worker connection.
- Confirm that the server recovers and serves a new request.

This is the hardest of the three goals because it touches SSH, worker
startup, ring rebuilds, active requests, and chat state.
