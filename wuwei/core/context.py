from wuwei.llm import Message, ToolCall


class Context:
    def __init__(self):
        self._messages:list[Message]=[]

    def add_user_message(self,content:str):
        self._messages.append(Message(role="user",content=content))

    def add_system_message(self,content:str):
        self._messages.append(Message(role="system",content=content))

    def add_tool_message(self,content:str,tool_call_id:str|None):
        self._messages.append(Message(role="tool",content=content,tool_call_id=tool_call_id))

    def add_ai_message(self,content:str,tool_calls:list[ToolCall]|None=None):
        self._messages.append(Message(role="assistant",content=content,tool_calls=tool_calls))

    def get_messages(self)->list[Message]:
        return self._messages

    def get_last_message(self)->Message|None:
        return self._messages[-1] if self._messages else None

    def reset(self):
        self._messages.clear()
