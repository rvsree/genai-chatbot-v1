def react_mermaid() -> str:
    return """flowchart TD
    U[User] -->|POST /react-single-agent/react/ask| R[ReAct Agent]
    R --> T[Tool Planner]
    T -->|vector_search| VS[RetrievalTools.vector_search]
    T -->|index_lookup| IL[RetrievalTools.index_lookup]
    T -->|get_chunk| GC[RetrievalTools.get_chunk]
    VS --> C1[Context]
    GC --> C2[Context]
    R --> OA[OpenAI Chat (final)]
    C1 --> OA
    C2 --> OA
    OA --> A[Answer+citations]
    """

def functions_mermaid() -> str:
    return """flowchart TD
    U[User] -->|POST /react-single-agent/functions/ask| F[Function-Calling]
    F --> OA1[OpenAI Chat (tool_calls)]
    OA1 -->|vector_search| VS[RetrievalTools.vector_search]
    VS --> TM1[tool message]
    TM1 --> OA2[OpenAI Chat (final)]
    OA1 -->|get_chunk (follow-up)| GC[RetrievalTools.get_chunk]
    GC --> TM2[tool message]
    TM2 --> OA2
    OA2 --> A[Answer+citations]
    """
