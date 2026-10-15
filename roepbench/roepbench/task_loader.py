from typing import Type


def load_task_class(task_name: str) -> Type:
    """Load supported RLBench task classes.

    Gate-1 supports only open_drawer.
    """
    normalized = task_name.strip().lower()
    if normalized != "open_drawer":
        raise ValueError(
            f"Unsupported task '{task_name}'. Gate-1 only supports 'open_drawer'."
        )

    # Explicit module import requested by spec.
    from rlbench.tasks.open_drawer import OpenDrawer

    return OpenDrawer
