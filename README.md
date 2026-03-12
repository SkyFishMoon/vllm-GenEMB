# vLLM with Special Token Hidden State Capture

This repository contains a modified version of [vLLM](https://github.com/vllm-project/vllm) that supports **capturing hidden states associated with a special token during generation**.

This functionality is designed for tasks where the model must **generate a representation (embedding) as part of the decoding process**, rather than producing it via a separate forward pass. The captured hidden state can be directly returned through the API together with the generated text.

This modification was developed for embedding-oriented generation workflows such as **Think-Then-Embed style pipelines**.

---

# Motivation

Many modern retrieval and reasoning pipelines require the model to:

1. Generate intermediate reasoning tokens
2. Emit a **special marker token**
3. Produce a **representation (embedding) derived from the model hidden state**

In standard vLLM inference, the hidden states used during decoding are **not exposed to the API**, making it difficult to obtain embeddings aligned with generation.

A naive approach would require:

* Running generation
* Running an additional forward pass over the full sequence
* Extracting the hidden state for the special token

This is inefficient and increases latency.

This repository modifies vLLM so that the **desired hidden state can be captured directly during generation**, without requiring an additional forward pass.

---

# Key Idea

When the model generates a designated **special token**, we capture the hidden state corresponding to the **next decoding step**.

This design choice is important.

### Why capture the next-step hidden state?

For decoder-only language models:

* The hidden state used to **predict the special token** does **not yet include that token in the context**.
* The hidden state of the **next step** corresponds to the representation **after the special token has been incorporated into the sequence**.

This behavior matches the hidden state that would be obtained from a standard forward pass over the final sequence.

Therefore, we capture:

```
hidden(state after special token enters context)
```

rather than:

```
hidden(state used to generate the special token)
```

---

# Features

The modified vLLM supports:

* Capturing hidden states when a **specified token ID appears in generation**
* Returning the captured hidden state through the API response
* Optional **L2 normalization** of the captured representation
* Automatic **fallback to the final token hidden state** if the special token occurs at the end of generation

The system remains fully compatible with:

* vLLM batching
* streaming generation
* speculative decoding
* async scheduling

---

# Implementation Overview

The main changes are implemented inside the vLLM generation pipeline.

### 1. Sampling Metadata Extension

We add two new fields to sampling metadata:

```
capture_token_ids
capture_token_hidden_normalize
```

These specify:

* which token should trigger hidden state capture
* whether the hidden state should be normalized

---

### 2. Request State Tracking

Each request tracks:

```
captured_hidden
last_token_hidden
pending_capture_after_special
```

These fields control when and how hidden states are captured.

---

### 3. Delayed Capture Logic

Instead of capturing hidden states immediately when the special token is generated, we:

1. Detect when the special token appears
2. Mark the request as `pending_capture_after_special`
3. Capture the hidden state **at the next decoding step**

This logic is implemented in:

```
_gpu_model_runner._bookkeeping_sync()
```

---

### 4. Hidden State Fallback

If the special token occurs at the **final decoding step**, the next-step hidden state does not exist.

In that case we fall back to:

```
last_token_hidden
```

which stores the hidden state of the most recent decoding step.

---

# API Output

The server response includes a new field:

```
captured_hidden
```

Example:

```json
{
  "token_id": 32000,
  "hidden": [0.13, -0.42, 0.98, ...]
}
```

where:

* `token_id` is the trigger token
* `hidden` is the captured hidden state vector

---

# Usage

Start the server as usual:

```bash
vllm serve <model-path> \
  --served-model-name my-model \
  --host 0.0.0.0 \
  --port 8000
```

Send a request that specifies the **capture token id**.

Example request:

```json
{
  "model": "my-model",
  "messages": [
    {"role": "user", "content": "Explain the concept briefly."}
  ],
  "capture_token_id": 32000
}
```

If the token appears in generation, the response will include the captured hidden state.

---

# Example Workflow

A typical embedding workflow:

```
Prompt
   ↓
Model reasoning
   ↓
<SPECIAL_TOKEN>
   ↓
Hidden state captured
   ↓
Returned as embedding
```

This enables generation and embedding extraction in **one decoding pass**.

---

# Compatibility

This modification is designed to be minimally invasive to the vLLM architecture.

The following components remain unchanged:

* scheduler
* KV cache management
* tensor parallelism
* streaming API
* batching system

---

# Limitations

* Only one capture token per request is currently supported.
* The first occurrence of the token is used.
* Hidden states are captured from the final transformer layer.

---

# Future Work

Potential extensions include:

* capturing multiple token positions
* exposing intermediate layer representations
* returning embeddings for multiple special tokens

---

# Acknowledgements

This project builds on the excellent work of the **vLLM team**.

Original repository:

[https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
