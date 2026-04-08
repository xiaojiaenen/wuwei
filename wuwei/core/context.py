from wuwei.llm import Message


class Context:
    def __init__(self):
        self._messages:list[Message]=[]

    def add_user_message(self,content:str):
        self._messages.append(Message(role="user",content=content))

    def add_system_message(self,content:str):
        self._messages.append(Message(role="system",content=content))

    def add_tool_message(self,content:str,tool_call_id:str|None):
        self._messages.append(Message(role="tool",content=content,tool_call_id=tool_call_id))

    def add_tool_call(self,content:str,tool_call_id:str|None):
        self._messages.append(Message(role="tool_call",content=content,tool_call_id=tool_call_id))