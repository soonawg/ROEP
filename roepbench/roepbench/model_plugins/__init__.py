"""External-model plugin templates for ROEP model adapters.

Each plugin module must expose:
    build_adapter(checkpoint_path: str, device: str, model_name: str) -> object

Returned object contract:
    - has method: predict_action(obs_dict: dict, lang_goal: Optional[str]) -> action
    - action should be array-like and will be normalized by rlbench_runner.
"""

