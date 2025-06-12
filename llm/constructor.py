from typing import Any

from pydantic import BaseModel


class BaseLLM(BaseModel):
    model: str
    api_key: str | None = None
    credentials: str | None = None
    scope: str | None = None
    verify_ssl_certs: bool | None = None
    profanity_check: bool | None = None

    def to_dict(self, exclude_none: bool = True) -> dict[str, Any]:
        return self.model_dump(exclude_none=exclude_none)
