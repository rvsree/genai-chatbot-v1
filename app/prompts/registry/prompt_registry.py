from dataclasses import dataclass

@dataclass(frozen=True)
class PromptBundle:
    system: str
    user_template: str

class PromptRegistry:
    def __init__(self, react_bundle: PromptBundle, func_bundle: PromptBundle):
        self.react = react_bundle
        self.func = func_bundle
