from fastapi import HTTPException
from pydantic import BaseModel, Field

from core.src.generate import RuntimeModel

model = RuntimeModel()


class Context(BaseModel):
    prompt: str = Field(max_length=1000)
    max_tokens: int = Field(default=500, ge=1, le=2000)

    def response(self) -> dict:
        try:
            story = model.request(self.prompt, self.max_tokens)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return {"response": story}
