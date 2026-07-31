from tesseractcli.tools import (
    edit_file,
    exec_tool,
    list_directory,
    read_file,
    write_file,
)
from tesseractcli.tools.registry import ToolRegistry


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    write_file.register(registry)
    read_file.register(registry)
    edit_file.register(registry)
    exec_tool.register(registry)
    list_directory.register(registry)
    return registry
