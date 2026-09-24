# 7. Generation & Prompting

The generation layer is where the LLM takes the retrieved and reranked context and produces a natural language answer. This chapter covers the LLM client, context construction, and prompt engineering.

**Code**: `src/generation/llm.py`, `src/generation/context_builder.py`, `src/generation/prompts.py`

---

## 7.1 The LLM Client

AutoRAG supports three LLM providers through a unified interface:

| Provider | Example Models | API Style | Cost |
|---|---|---|---|
| **OpenAI** | GPT-4o, GPT-4o-mini | Cloud API | Pay per token |
| **Google Gemini** | Gemini 1.5 Flash/Pro | Cloud API | Pay per token |
| **Ollama** | Llama 3.2, Mistral | Local HTTP | Free (your hardware) |

### Unified Interface

All three providers are accessed through the same `LLMClient` class:

```python
class LLMClient:
    def __init__(self, provider, model_name, temperature, max_tokens):
        if provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == "gemini":
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        elif provider == "ollama":
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    def generate(self, prompt, system_prompt=None) -> LLMResponse:
        # Same interface regardless of provider
        # Returns: LLMResponse(text, model, input_tokens, output_tokens, latency_ms)
```

### Why Ollama Uses OpenAI's Client

Ollama exposes an **OpenAI-compatible** API endpoint at `localhost:11434/v1`. This means we can use the exact same OpenAI Python SDK to talk to local models. We just change the `base_url`. This is a common pattern in the industry.

### Temperature

Temperature controls the randomness of the LLM's output:

```
temperature = 0.0  → Deterministic (always picks the most probable token)
temperature = 0.7  → Moderate creativity
temperature = 1.0  → Full randomness
```

We use **temperature = 0.0** for RAG because we want consistent, factual answers. Creative writing tasks would use higher temperatures.

**Code**: See `src/generation/llm.py`

---

## 7.2 Context Construction

### The Problem

After reranking, we have ~5 `ScoredChunk` objects. We need to convert these into a single string that the LLM can read as its "reference material." This is not as trivial as it sounds:

1. **Deduplication**: Two chunks might have significant text overlap (from chunk overlapping during ingestion).
2. **Ordering**: Should chunks be ordered by relevance score? Or by their original position in the document?
3. **Token budget**: The LLM's context window is finite. We can't paste in unlimited text.

### Context Builder Pipeline

```python
class ContextBuilder:
    def build(self, query, scored_chunks) -> tuple[str, list[ScoredChunk]]:
        # Step 1: Deduplicate highly overlapping chunks
        deduped = self._deduplicate(scored_chunks)

        # Step 2: Truncate to fit token budget
        selected = self._select_within_budget(deduped)

        # Step 3: Order chunks
        ordered = self._order(selected)

        # Step 4: Format as a numbered context string
        context_str = self._format(ordered)

        return context_str, ordered
```

### Deduplication

Because of chunk overlap (e.g., 200-character overlap), two adjacent chunks in the original document might share significant text. If both are retrieved, we'd be wasting context window space on duplicated information.

We use **text overlap ratio** to detect near-duplicates:

```
                    2 × |overlap(A, B)|
overlap_ratio = ──────────────────────────
                    |A| + |B|
```

If `overlap_ratio > threshold` (default: 0.9), we keep only the higher-scored chunk.

### Ordering Strategies

| Strategy | When to Use |
|---|---|
| **Relevance** (default) | Most relevant chunk first. Best for factoid QA where the top chunk usually has the answer. |
| **Position** | Original document order. Better for summarization where narrative flow matters. |

### The Formatted Output

The context builder produces output like this that gets injected into the LLM prompt:

```
[1] (Source: quantumflow.md, Score: 0.94)
QuantumFlow Innovations was a pioneering quantum computing startup founded
in 2018 by Dr. Aris Thorne and Dr. Elena Rostova...

[2] (Source: nexgen.md, Score: 0.87)
NexGen Tech is a multinational technology conglomerate headquartered in
Austin, Texas...
```

**Code**: See `src/generation/context_builder.py`

---

## 7.3 Prompt Engineering

Prompts are the instructions we give to the LLM. Well-designed prompts are **critical** — the same model can give wildly different answers based on how you ask.

### RAG Generation Prompt

This is the main prompt used for answering questions with retrieved context:

```
SYSTEM:
You are a helpful assistant that answers questions based on the provided context.
- Only use information from the context to answer
- If the context doesn't contain enough information, say so
- Cite your sources by referring to the context numbers [1], [2], etc.
- Be concise and direct

USER:
Context:
[1] (Source: quantumflow.md) QuantumFlow Innovations was founded in 2018...
[2] (Source: nexgen.md) NexGen Tech acquired QuantumFlow for $1.2B...

Question: Who acquired QuantumFlow Innovations?
```

### Key Prompt Design Principles

1. **Grounding instruction**: "Only use information from the context" — this reduces hallucination by telling the LLM to stay grounded in the evidence.

2. **Failure mode**: "If the context doesn't contain enough information, say so" — better to say "I don't know" than to hallucinate.

3. **Source citation**: "Cite your sources by referring to [1], [2]" — enables verifiability.

4. **System vs User prompt**: The system prompt sets the persona and rules. The user prompt contains the actual query and context. This separation lets us reuse the same system prompt across all queries.

### Query Classification Prompt

Used by the Adaptive Router to decide the retrieval strategy:

```
SYSTEM:
Classify the complexity of the following question into one of three categories:
A) DIRECT - Simple factual question answerable from general knowledge
B) SINGLE_HOP - Requires looking up information from one source
C) MULTI_HOP - Requires connecting information from multiple sources

Respond with only the letter: A, B, or C.

USER:
Question: Who is the CEO of the company that acquired QuantumFlow?
```

The LLM should respond with just "C" because answering requires two steps (which company acquired QuantumFlow → who is the CEO of that company).

### Query Decomposition Prompt

Used by the Multi-Hop Pipeline to break complex questions into sub-questions:

```
SYSTEM:
Break the following complex question into simpler sub-questions that can
each be answered independently. Return one sub-question per line.

USER:
Question: Who is the CEO of the company that acquired QuantumFlow Innovations?
```

Expected LLM output:
```
1. Which company acquired QuantumFlow Innovations?
2. Who is the CEO of that company?
```

### Multi-Hop Synthesis Prompt

After all sub-questions are answered, this prompt synthesizes the final answer:

```
SYSTEM:
You are given a complex question and the answers to its sub-questions.
Synthesize a complete, well-structured answer to the original question.

USER:
Original question: Who is the CEO of the company that acquired QuantumFlow?

Evidence:
- Sub-question: Which company acquired QuantumFlow Innovations?
  Answer: NexGen Tech acquired QuantumFlow Innovations in March 2024.

- Sub-question: Who is the CEO of NexGen Tech?
  Answer: Sarah Jenkins is the CEO of NexGen Tech.

Provide the final synthesized answer:
```

### Faithfulness Judge Prompt

Used for evaluation (LLM-as-Judge) to check if an answer is grounded in context:

```
SYSTEM:
Evaluate whether the answer is faithfully supported by the context.
Rate on a scale of 1-5 and provide reasoning.
Respond in the format:
SCORE: [1-5]
REASONING: [your explanation]

USER:
Question: Who founded QuantumFlow?
Context: QuantumFlow was founded in 2018 by Dr. Aris Thorne and Dr. Elena Rostova.
Answer: Dr. Aris Thorne and Dr. Elena Rostova founded QuantumFlow in 2018.
```

**Code**: See `src/generation/prompts.py` for all prompt templates.

---

## 7.4 Token Budget Management

### The Problem

LLMs have finite context windows:

| Model | Context Window |
|---|---|
| Llama 3.2 (3B) | 128K tokens |
| GPT-4o-mini | 128K tokens |
| Gemini 1.5 Flash | 1M tokens |

Even with large windows, more context = more cost + more latency. We want to include only the most relevant information.

### Our Budget Strategy

```yaml
context:
  max_contexts: 5          # Maximum chunks to include
  max_context_tokens: 3000 # Maximum total tokens for context
```

The context builder selects chunks greedily by score until either `max_contexts` or `max_context_tokens` is reached, whichever comes first.

---

## 7.5 Interview-Ready Talking Points

> **Q: Why use a system prompt separate from the user prompt?**
> A: System prompts set persistent behavioral rules (grounding, citation format) that apply to every query. User prompts contain the variable content (context + question). This separation enables consistent behavior across different queries without repeating instructions.

> **Q: How do you prevent hallucination in RAG?**
> A: Three mechanisms: (1) The system prompt explicitly instructs the LLM to only use the provided context. (2) We include the actual source text so the LLM can copy/paraphrase rather than invent. (3) We use faithfulness evaluation (LLM-as-Judge) to detect when the model strays from the evidence.

> **Q: Why temperature = 0 for RAG?**
> A: Temperature controls randomness. For factual QA, we want deterministic, reproducible answers. Temperature = 0 means the model always picks the highest-probability token, minimizing creative deviation from the evidence.

---

**Next**: [Chapter 8 — Adaptive RAG →](08_adaptive_rag.md)
