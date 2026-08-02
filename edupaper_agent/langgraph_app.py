"""LangGraph Studio entry point.

The FastAPI application uses the same compiled graph through ServiceContainer.
"""

from .config import get_settings
from .services import ServiceContainer

_services = ServiceContainer(get_settings())
graph = _services.graph
