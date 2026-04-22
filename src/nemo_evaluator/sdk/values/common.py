"""Common value types used throughout evaluator SDK runtime."""

# Migrated from: services/evaluator/src/nmp/evaluator/app/values/common.py

from enum import Enum

from pydantic import Field, RootModel


class SupportedJobTypes(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    RETRIEVER = "retriever"


class SecretRef(RootModel):
    root: str = Field(
        description="Reference to a secret. Format: 'secret_name' (uses request workspace) or 'workspace/secret_name' (explicit workspace).",
        pattern=r"^[a-z0-9_-]+(/[a-z0-9_-]+)?$",
        examples=[
            "my-secret",
            "my-workspace/my-secret",
        ],
    )
