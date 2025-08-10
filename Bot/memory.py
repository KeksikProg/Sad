from langgraph.checkpoint.memory import MemorySaver

class CappingMemorySaver(MemorySaver):
    def __init__(self, max_messages: int = 15):
        super().__init__()
        self.max_messages = max_messages

    def put(self, config, checkpoint, metadata=None, new_versions=None):
        ch = checkpoint.get("channel_values", {})
        msgs = ch.get("messages", [])
        
        pinned = [m for m in msgs if getattr(m, "type", None) == "system"]
        tail   = [m for m in msgs if getattr(m, "type", None) != "system"]

        if len(tail) > self.max_messages:
            tail = tail[-self.max_messages:]

        ch["messages"] = pinned + tail
        checkpoint["channel_values"] = ch
        return super().put(config, checkpoint, metadata, new_versions)