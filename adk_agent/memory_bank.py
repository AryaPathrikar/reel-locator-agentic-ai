# adk_agent/memory_bank.py

class MemoryBank:
    """
    Simple custom long-term memory simulation.
    Stores user-specific text and produces a compacted summary.
    """

    def __init__(self):
        self.entries = []

    def store(self, text: str):
        """Store any meaningful user or system message."""
        self.entries.append(text)

    def compact(self) -> str:
        """
        Compact memory for context engineering.
        We keep only the last few entries and join them.
        """
        if not self.entries:
            return ""

        # Compact the last 5 memories
        recent = self.entries[-5:]
        return " || ".join(recent)
