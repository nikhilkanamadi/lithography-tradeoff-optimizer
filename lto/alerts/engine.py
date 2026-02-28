"""Alert engine — evaluates conditions and dispatches alerts.

Provides configurable, per-job-class alert rules with 4 severity levels.
Includes deduplication, cooldown, and alert quality tracking.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from uuid import uuid4

from lto.schemas import AlertInfo, AlertSeverity, SimulationResult

logger = logging.getLogger(__name__)


class AlertRule:
    """A single configurable alert rule."""

    def __init__(
        self,
        name: str,
        condition_fn,
        severity: AlertSeverity,
        message_template: str,
        recommendation: str = "",
        cooldown_seconds: int = 300,
        job_classes: list[int] | None = None,
    ):
        self.name = name
        self.condition_fn = condition_fn
        self.severity = severity
        self.message_template = message_template
        self.recommendation = recommendation
        self.cooldown_seconds = cooldown_seconds
        self.job_classes = job_classes  # None = all classes
        self._last_fired: Optional[datetime] = None

    def evaluate(self, result: SimulationResult) -> Optional[AlertInfo]:
        """Evaluate the rule against a simulation result.

        Returns AlertInfo if the rule fires, None otherwise.
        """
        # Job class filter
        if self.job_classes and result.parameters.job_class not in self.job_classes:
            return None

        # Cooldown check
        if self._last_fired and (
            datetime.utcnow() - self._last_fired < timedelta(seconds=self.cooldown_seconds)
        ):
            return None

        # Evaluate condition
        if not self.condition_fn(result):
            return None

        self._last_fired = datetime.utcnow()

        return AlertInfo(
            severity=self.severity,
            job_id=result.job_id,
            alert_type=self.name,
            message=self.message_template.format(result=result),
            recommendation=self.recommendation,
        )


class AlertEngine:
    """Main alert engine — manages rules and dispatches alerts."""

    def __init__(self):
        self.rules: list[AlertRule] = []
        self.channels: list[Any] = []  # Alert channels (Slack, email, etc.)
        self.history: list[AlertInfo] = []
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Register the default alert rules per ARCHITECTURE.md spec."""

        self.rules.extend([
            AlertRule(
                name="critical_yield_risk",
                condition_fn=lambda r: r.tradeoff_signals.yield_risk > 0.8,
                severity=AlertSeverity.CRITICAL,
                message_template=(
                    "CRITICAL: Yield risk {result.tradeoff_signals.yield_risk:.3f} "
                    "exceeds 0.8 threshold for job {result.job_id}"
                ),
                recommendation="Reduce NA or increase dose to improve yield margin",
                cooldown_seconds=60,
            ),
            AlertRule(
                name="emergency_yield_risk",
                condition_fn=lambda r: r.tradeoff_signals.yield_risk > 0.95,
                severity=AlertSeverity.EMERGENCY,
                message_template=(
                    "🚨 EMERGENCY: Yield risk {result.tradeoff_signals.yield_risk:.3f} "
                    "near 1.0 — immediate intervention required for job {result.job_id}"
                ),
                recommendation="HALT production immediately. Review all parameters.",
                cooldown_seconds=10,
            ),
            AlertRule(
                name="low_overall_health",
                condition_fn=lambda r: r.tradeoff_signals.overall_health < 0.4,
                severity=AlertSeverity.WARN,
                message_template=(
                    "Low overall health {result.tradeoff_signals.overall_health:.3f} "
                    "for job {result.job_id}"
                ),
                recommendation="Review tradeoff configuration — multiple dimensions degraded",
                cooldown_seconds=120,
            ),
            AlertRule(
                name="surrogate_drift",
                condition_fn=lambda r: (
                    r.parameters.use_ai_surrogate
                    and r.tradeoff_signals.surrogate_reliability < 0.90
                ),
                severity=AlertSeverity.WARN,
                message_template=(
                    "Surrogate reliability {result.tradeoff_signals.surrogate_reliability:.3f} "
                    "below 0.90 for job {result.job_id}"
                ),
                recommendation="Consider running full physics simulation for validation",
                cooldown_seconds=300,
            ),
            AlertRule(
                name="slow_compute",
                condition_fn=lambda r: r.outputs.compute_time_s > 60.0,
                severity=AlertSeverity.INFO,
                message_template=(
                    "Slow simulation: {result.outputs.compute_time_s:.1f}s "
                    "for job {result.job_id}"
                ),
                recommendation="Consider coarser grid or AI surrogate for faster results",
                cooldown_seconds=600,
            ),
            AlertRule(
                name="poor_resolution_dof_tradeoff",
                condition_fn=lambda r: r.tradeoff_signals.resolution_vs_dof < 0.3,
                severity=AlertSeverity.WARN,
                message_template=(
                    "Resolution vs DoF tradeoff severely degraded "
                    "({result.tradeoff_signals.resolution_vs_dof:.3f}) for job {result.job_id}"
                ),
                recommendation="Reduce NA to improve depth of focus, or accept resolution loss",
                cooldown_seconds=180,
            ),
        ])

    def add_rule(self, rule: AlertRule) -> None:
        """Add a custom alert rule."""
        self.rules.append(rule)

    def add_channel(self, channel) -> None:
        """Register an alert notification channel (Slack, email, etc.)."""
        self.channels.append(channel)

    def evaluate(self, result: SimulationResult) -> list[AlertInfo]:
        """Evaluate all rules against a simulation result.

        Args:
            result: Completed simulation result.

        Returns:
            List of fired alerts (may be empty).
        """
        fired = []
        for rule in self.rules:
            alert = rule.evaluate(result)
            if alert:
                fired.append(alert)
                self.history.append(alert)
                logger.info(
                    f"Alert fired: [{alert.severity.value}] {alert.alert_type} — {alert.message}"
                )

        # Dispatch to channels
        if fired:
            self._dispatch(fired)

        return fired

    def _dispatch(self, alerts: list[AlertInfo]) -> None:
        """Send alerts to all registered channels."""
        for channel in self.channels:
            try:
                channel.send(alerts)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")

    def get_active_alerts(self, since_minutes: int = 60) -> list[AlertInfo]:
        """Get recent unacknowledged alerts."""
        cutoff = datetime.utcnow() - timedelta(minutes=since_minutes)
        return [
            a for a in self.history
            if a.fired_at > cutoff and a.acknowledged_at is None
        ]

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.history:
            if alert.alert_id == alert_id:
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.utcnow()
                return True
        return False
