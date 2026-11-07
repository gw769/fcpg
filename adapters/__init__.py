"""Adapter registry and loader.

Adapters implement the interfaces defined in interfaces/ for specific hardware
and algorithm combinations.  New adapters are registered by adding an entry
to a serve config YAML, not by modifying this file.

Usage
-----
from adapters import load_adapter

bus = load_adapter("robot_bus", config["robot"]["bus"])
"""

from __future__ import annotations

import importlib
from typing import Any


def load_adapter(interface_name: str, adapter_config: dict[str, Any]) -> Any:
    """Instantiate an adapter from config.

    Config must contain:
        adapter_class: fully qualified Python class path
                       e.g. "adapters.so101.bus.FeetechBusAdapter"
    All other keys are forwarded to the adapter constructor as kwargs.

    Example config (YAML)::

        robot:
          bus:
            adapter_class: adapters.so101.bus.FeetechBusAdapter
            port: /dev/ttyUSB0
            baudrate: 1000000
            motor_ids: [1, 2, 3, 4, 5, 6]
    """
    if "adapter_class" not in adapter_config:
        raise ValueError(
            f"adapter config for '{interface_name}' must contain 'adapter_class'"
        )
    class_path: str = adapter_config["adapter_class"]
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    kwargs = {k: v for k, v in adapter_config.items() if k != "adapter_class"}
    return cls(**kwargs)
