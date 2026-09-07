"""The Dual-Agent Supervisor: secure maker-checker system with provider failover."""
from .core import CheckerVerdict, DualAgentSupervisor, ProviderOutage, make_failover

__all__ = ["CheckerVerdict", "DualAgentSupervisor", "ProviderOutage", "make_failover"]
