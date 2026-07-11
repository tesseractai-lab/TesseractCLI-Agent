
from tesseractcli.tools import write_file, read_file, edit_file,exec_tool,list_directory
from tesseractcli.tools.registry import ToolRegistry

def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    write_file.register(registry)
    read_file.register(registry)
    edit_file.register(registry)
    exec_tool.register(registry)
    list_directory.register(registry)
    return registry
