from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The message to screen")


class FirewallResponse(BaseModel):
    safe: bool
    threat: Optional[str] = None
    layer: Optional[str] = None
