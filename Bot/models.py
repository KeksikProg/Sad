from pydantic import BaseModel

class Input(BaseModel):
    message: str
    user_id: str

class CodesIn(BaseModel):
    codes: list[str]