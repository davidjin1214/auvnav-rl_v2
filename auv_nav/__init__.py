"""AUV wake-navigation RL framework."""

from .vehicle import Remus100, S, ssa, sat, ActuatorState, VehicleParams, rk4_step, ValidityReport
from .flow import WakeField, FlowSampler, ReferenceFlowEstimator, WakeFieldMetadata, rotation_world_to_body, rotation_body_to_world
from .autopilot import (
    DepthHold6DOFBackend, DepthHold6DOFBackendConfig,
    EquivalentCurrentModel, HeadingAutopilotConfig, DepthHoldAutopilotConfig,
)
from .env import (
    PlanarRemusEnv, PlanarRemusEnvConfig,
    ObservationLayout, ObsNormScales,
    TaskSampler, TaskSamplerConfig,
    ObservationHistoryWrapper,
)
from .reward import RewardModel, SafetyCostModel, RewardModelConfig, SafetyCostModelConfig
from .networks import MLP, require_torch
from .replay import TransitionReplay, TransitionReplayConfig
from .sac import SACAgent, SACConfig
from .baselines import (
    GoalSeekPolicy,
    CrossCurrentCompensationPolicy,
    StillWaterStraightLine,
    WorldFrameCurrentCompensationPolicy,
    PrivilegedCorridorPolicy,
)

__all__ = [
    "Remus100", "S", "ssa", "sat", "ActuatorState", "VehicleParams", "rk4_step", "ValidityReport",
    "WakeField", "FlowSampler", "ReferenceFlowEstimator", "WakeFieldMetadata",
    "rotation_world_to_body", "rotation_body_to_world",
    "DepthHold6DOFBackend", "DepthHold6DOFBackendConfig",
    "EquivalentCurrentModel", "HeadingAutopilotConfig", "DepthHoldAutopilotConfig",
    "PlanarRemusEnv", "PlanarRemusEnvConfig",
    "ObservationLayout", "ObsNormScales",
    "TaskSampler", "TaskSamplerConfig",
    "ObservationHistoryWrapper",
    "RewardModel", "SafetyCostModel", "RewardModelConfig", "SafetyCostModelConfig",
    "MLP", "require_torch",
    "TransitionReplay", "TransitionReplayConfig",
    "SACAgent", "SACConfig",
    "GoalSeekPolicy", "CrossCurrentCompensationPolicy", "StillWaterStraightLine",
    "WorldFrameCurrentCompensationPolicy", "PrivilegedCorridorPolicy",
]
