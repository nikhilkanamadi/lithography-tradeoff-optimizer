"""Slack webhook alert channel.

Sends formatted alert cards to a Slack channel via incoming webhook.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional
from urllib.request import Request, urlopen

from lto.schemas import AlertInfo, AlertSeverity

logger = logging.getLogger(__name__)

SEVERITY_EMOJI = {
    AlertSeverity.INFO: "ℹ️",
    AlertSeverity.WARN: "⚠️",
    AlertSeverity.CRITICAL: "🔴",
    AlertSeverity.EMERGENCY: "🚨",
}

SEVERITY_COLOR = {
    AlertSeverity.INFO: "#36a64f",
    AlertSeverity.WARN: "#daa520",
    AlertSeverity.CRITICAL: "#ff4444",
    AlertSeverity.EMERGENCY: "#8b0000",
}


class SlackChannel:
    """Slack incoming webhook alert channel."""

    def __init__(self, webhook_url: str | None = None, channel: str = "#lto-alerts"):
        self.webhook_url = webhook_url or os.getenv("LTO_SLACK_WEBHOOK_URL", "")
        self.channel = channel

    def send(self, alerts: list[AlertInfo]) -> None:
        """Send alerts to Slack."""
        if not self.webhook_url:
            logger.debug("Slack webhook URL not configured — skipping")
            return

        for alert in alerts:
            payload = self._format_alert(alert)
            try:
                req = Request(
                    self.webhook_url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                urlopen(req)
                logger.info(f"Alert sent to Slack: {alert.alert_type}")
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {e}")

    def _format_alert(self, alert: AlertInfo) -> dict:
        """Format an alert as a Slack Block Kit message."""
        emoji = SEVERITY_EMOJI.get(alert.severity, "📋")
        color = SEVERITY_COLOR.get(alert.severity, "#cccccc")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} LTO Alert: {alert.alert_type}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Severity:* {alert.severity.value.upper()}"},
                    {"type": "mrkdwn", "text": f"*Job:* {alert.job_id or 'N/A'}"},
                    {"type": "mrkdwn", "text": f"*Time:* {alert.fired_at.isoformat()}"},
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": alert.message},
            },
        ]

        if alert.recommendation:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*💡 Recommendation:* {alert.recommendation}",
                },
            })

        return {
            "channel": self.channel,
            "attachments": [{"color": color, "blocks": blocks}],
        }
