"""Registry helpers for offline RL algorithms."""

from __future__ import annotations

from typing import Any


def normalize_offline_algo(algo: str | None) -> str:
    value = "td3bc" if algo is None else str(algo).strip().lower()
    if value not in {"td3bc", "rebrac", "fql"}:
        raise ValueError(f"Unsupported offline algorithm: {algo!r}")
    return value


def default_offline_save_dir(algo: str) -> str:
    algo_name = normalize_offline_algo(algo)
    return f"checkpoints/offline/{algo_name}"


def make_agent_config(algo: str, **kwargs: Any) -> Any:
    algo_name = normalize_offline_algo(algo)
    if algo_name == "td3bc":
        from .td3bc import TD3BCConfig

        return TD3BCConfig(**kwargs)

    if algo_name == "fql":
        from .fql import FQLConfig

        return FQLConfig(**kwargs)

    from .rebrac import ReBRACConfig

    return ReBRACConfig(**kwargs)


def make_agent(
    algo: str,
    config: Any,
    *,
    obs_normalizer: Any | None = None,
    device: str = "cpu",
) -> Any:
    algo_name = normalize_offline_algo(algo)
    if algo_name == "td3bc":
        from .td3bc import TD3BCAgent

        return TD3BCAgent(
            config,
            obs_normalizer=obs_normalizer,
            device=device,
        )

    if algo_name == "fql":
        from .fql import FQLAgent

        return FQLAgent(
            config,
            obs_normalizer=obs_normalizer,
            device=device,
        )

    from .rebrac import ReBRACAgent

    return ReBRACAgent(
        config,
        obs_normalizer=obs_normalizer,
        device=device,
    )


def make_agent_from_config_dict(
    algo: str,
    config_dict: dict[str, Any],
    *,
    obs_normalizer: Any | None = None,
    device: str = "cpu",
) -> Any:
    config = make_agent_config(algo, **config_dict)
    return make_agent(
        algo,
        config,
        obs_normalizer=obs_normalizer,
        device=device,
    )


def policy_from_payload(
    payload: dict[str, Any],
    *,
    device: str = "cpu",
) -> Any:
    algo_name = normalize_offline_algo(payload.get("algo"))
    if algo_name == "td3bc":
        from .td3bc import TD3BCPolicy

        return TD3BCPolicy.from_payload(payload, device=device)

    if algo_name == "fql":
        from .fql import FQLPolicy

        return FQLPolicy.from_payload(payload, device=device)

    from .rebrac import ReBRACPolicy

    return ReBRACPolicy.from_payload(payload, device=device)
