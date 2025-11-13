# --- Проверяет импорт библиотек AutoML
from __future__ import annotations

import importlib
import sys
from typing import Tuple

TARGET_MODULES = ("lightautoml", "fedot")


# --- Проверяет возможность импорта модуля
def check_module(module_name: str) -> Tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, ""
    except Exception as error:
        return False, str(error)


# --- Точка входа проверки AutoML библиотек
def main() -> int:
    failed = []
    for module in TARGET_MODULES:
        ok, info = check_module(module)
        if ok:
            print(f"[OK] {module} импортирован.")
        else:
            print(f"[FAIL] {module} не импортируется: {info}")
            failed.append(module)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

