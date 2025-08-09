from langgraph.checkpoint.memory import MemorySaver

class CappingMemorySaver(MemorySaver):
    def __init__(self, max_messages: int = 15):
        super().__init__()
        self.max_messages = max_messages

    def put(self, config, checkpoint, metadata=None, new_versions=None):
        # у prebuilt-агента сообщения лежат тут:
        ch = checkpoint.get("channel_values", {})
        msgs = ch.get("messages", [])
        # msgs — список BaseMessage; режем хвост, если нужно
        if isinstance(msgs, list) and len(msgs) > self.max_messages:
            ch["messages"] = msgs[-self.max_messages:]
            checkpoint["channel_values"] = ch
        # пробрасываем дальше
        return super().put(config, checkpoint, metadata, new_versions)