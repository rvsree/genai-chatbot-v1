# Extract “AS IS” templates from the sample; placeholders kept minimal.
REACT_SYSTEM = """You are a helpful financial analysis agent. Use tools carefully. Think step-by-step before acting. If a tool result answers the question, stop and summarize with citations."""
REACT_USER_TEMPLATE = """Question: {question}
Constraints:
- Prefer relevant filings and chunks.
- Cite parent ids like [tesla-2023].
Respond with a concise final answer once you have enough evidence."""

# Optional helper for thought formatting
REACT_THOUGHT_PREFIX = "Thought:"
REACT_ACTION_PREFIX = "Action:"
REACT_OBSERVATION_PREFIX = "Observation:"
