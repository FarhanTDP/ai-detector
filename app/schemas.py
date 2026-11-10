"""Request/response models for the JSON API."""

from __future__ import annotations

from pydantic import BaseModel, Field

MAX_TITLE = 200


class DocumentCreate(BaseModel):
    title: str = Field(default="Untitled document", max_length=MAX_TITLE)


class DocumentUpdate(BaseModel):
    title: str = Field(max_length=MAX_TITLE)
    content: str = Field(description="Quill Delta serialised as JSON")
    plain_text: str
    version: int = Field(ge=1, description="Version this edit is based on")
    force: bool = Field(default=False, description="Overwrite even if the version is stale")


class ScanRequest(BaseModel):
    text: str
    document_id: int | None = None
