import uuid
def new_id() -> str:
    return f"text-{uuid.uuid4().hex}"
