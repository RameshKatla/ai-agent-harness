from dataclasses import dataclass
from secrets import compare_digest
from typing import Protocol

from pydantic import SecretStr


@dataclass(frozen=True)
class Principal:
    user_id: str
    permissions: frozenset[str]


class Authenticator(Protocol):
    @property
    def configured(self) -> bool: ...

    async def authenticate(self, token: str) -> Principal | None: ...


class ConfiguredBearerAuthenticator:
    def __init__(self, token: SecretStr | None, principal: Principal) -> None:
        self._token = token
        self._principal = principal

    @property
    def configured(self) -> bool:
        return bool(self._token and self._token.get_secret_value())

    async def authenticate(self, token: str) -> Principal | None:
        configured = self._token.get_secret_value() if self._token else ""
        if configured and compare_digest(token.encode(), configured.encode()):
            return self._principal
        return None
