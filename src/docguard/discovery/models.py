"""Strict API inputs for explicit-source discovery and snapshot approval."""

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from docguard.discovery.fetch import validate_url

Text = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Reply(StrictInput):
    urls: list[Text] = Field(default_factory=list, max_length=20)
    message: Text | None = None
    version: Text | None = None
    product: Text | None = None
    allow_unversioned: bool = False

    @model_validator(mode="after")
    def not_empty(self):
        if not (
            self.urls
            or self.message
            or self.version
            or self.product
            or self.allow_unversioned
        ):
            raise ValueError("Provide URLs or clarification")
        return self

    @field_validator("urls")
    @classmethod
    def valid_urls(cls, values):
        return sorted({validate_url(url) for url in values})


class CreateTask(StrictInput):
    document_type: Literal["user_guide", "api_reference", "getting_started", "topic"] = "user_guide"
    goal: (
        Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)] | None
    ) = None
    product: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
    ]
    version: (
        Annotated[
            str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
        ]
        | None
    ) = None
    urls: list[Text] = Field(default_factory=list, max_length=20)
    official_only: bool = True

    @field_validator("product")
    @classmethod
    def required_product(cls, value):
        if value.casefold() == "unspecified":
            raise ValueError("A product name is required")
        return value

    @model_validator(mode="after")
    def default_goal(self):
        if not self.goal:
            target = self.product + (" " + self.version if self.version else "")
            self.goal = f"查找 {target} 的官方文档"
        return self

    @field_validator("urls")
    @classmethod
    def valid_urls(cls, values):
        return sorted({validate_url(url) for url in values})


class Approval(StrictInput):
    import_scope: Literal["pages", "collection"] = "pages"
    candidate_ids: list[
        Annotated[str, StringConstraints(min_length=1, max_length=100)]
    ] = Field(min_length=1, max_length=20)
    revision: int = Field(strict=True, ge=1)
    knowledge_base_id: Identifier
    idempotency_key: Annotated[
        str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    ]


class KnowledgeQuery(StrictInput):
    question: Text
    generate: bool = Field(default=False, strict=True)
