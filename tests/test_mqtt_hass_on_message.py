"""Regression tests for ledfx/integrations/mqtt_hass.py on_message handling.

Covers the HA light entity on/off toggle path: when Home Assistant sends a
bare `{"state": "on"}` or `{"state": "off"}` payload (no color, no effect)
on the entity's `set` topic, `virtual.active` must be updated.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ledfx.integrations import mqtt_hass
from ledfx.integrations.mqtt_hass import MQTT_HASS


def _build_msg(virtualid, payload_obj):
    msg = SimpleNamespace()
    msg.topic = f"homeassistant/light/{virtualid}/set"
    msg.payload = json.dumps(payload_obj).encode("utf-8")
    return msg


def _build_handler_stub(virtual, monkeypatch):
    """Build the minimum surface MQTT_HASS.on_message reads from `self`."""
    monkeypatch.setattr(mqtt_hass, "save_config", lambda **kwargs: None)

    virtuals = MagicMock()
    virtuals._paused = False
    virtuals.values.return_value = []
    virtuals.get = lambda vid, default=None: (
        virtual if vid == virtual.id else default
    )

    devices = MagicMock()
    devices.values.return_value = []

    ledfx = SimpleNamespace(
        virtuals=virtuals,
        devices=devices,
        config={},
        config_dir="/tmp",
        port=8888,
    )

    stub = SimpleNamespace(
        _ledfx=ledfx,
        _config={"topic": "homeassistant"},
        TRANSITION_MAPPING=MQTT_HASS.TRANSITION_MAPPING,
        publish_virtual_config=lambda *a, **kw: None,
        publish_virtual_paused=lambda *a, **kw: None,
    )
    return stub


def _build_virtual(virtualid="testvirtual", initial_active=False):
    return SimpleNamespace(
        id=virtualid,
        active=initial_active,
        virtual_cfg={},
        config={"name": virtualid, "icon_name": "mdi:led-strip"},
        active_effect=None,
        set_effect=MagicMock(),
        update_config=MagicMock(),
    )


@pytest.mark.parametrize(
    "state_value,expected_active",
    [("on", True), ("off", False)],
)
def test_state_only_payload_updates_virtual_active(
    state_value, expected_active, monkeypatch
):
    """HA bare on/off toggle: payload has only `state`, no color or effect."""
    virtual = _build_virtual(initial_active=not expected_active)
    stub = _build_handler_stub(virtual, monkeypatch)
    msg = _build_msg(virtual.id, {"state": state_value})

    MQTT_HASS.on_message(stub, client=MagicMock(), userdata=None, msg=msg)

    assert virtual.active is expected_active
    assert virtual.virtual_cfg["active"] is expected_active


def test_state_with_color_still_updates_virtual_active(monkeypatch):
    """Color path regression: color + state together must still flip active."""
    virtual = _build_virtual(initial_active=False)
    stub = _build_handler_stub(virtual, monkeypatch)
    effects = MagicMock()
    effects.create.return_value = MagicMock()
    effects.classes.return_value = {}
    stub._ledfx.effects = effects

    msg = _build_msg(virtual.id, {"state": "on", "color": [255, 0, 0]})

    MQTT_HASS.on_message(stub, client=MagicMock(), userdata=None, msg=msg)

    assert virtual.active is True
    virtual.set_effect.assert_called_once()


def test_state_off_with_color_deactivates(monkeypatch):
    """A color payload with state=off should deactivate, not activate."""
    virtual = _build_virtual(initial_active=True)
    stub = _build_handler_stub(virtual, monkeypatch)
    effects = MagicMock()
    effects.create.return_value = MagicMock()
    effects.classes.return_value = {}
    stub._ledfx.effects = effects

    msg = _build_msg(virtual.id, {"state": "off", "color": [255, 0, 0]})

    MQTT_HASS.on_message(stub, client=MagicMock(), userdata=None, msg=msg)

    assert virtual.active is False


class _PropertyVirtual:
    """Virtual stub whose `active` setter mimics Virtual.activate()'s
    refusal to start without a configured effect.

    Mirrors the real `Virtual.activate` guard at `ledfx/virtuals.py:944-947`:
    setting `active = True` raises RuntimeError when no effect is set;
    `active = False` is always safe.
    """

    def __init__(self, virtualid="testvirtual", has_effect=False):
        self.id = virtualid
        self.virtual_cfg = {}
        self.config = {"name": virtualid, "icon_name": "mdi:led-strip"}
        self.active_effect = None
        self.set_effect = MagicMock()
        self.update_config = MagicMock()
        self._active = False
        self._has_effect = has_effect

    @property
    def active(self):
        return self._active

    @active.setter
    def active(self, value):
        value = bool(value)
        if value and not self._has_effect:
            raise RuntimeError(
                f"Virtual {self.id}: Cannot activate, no configured effect"
            )
        self._active = value


def test_state_on_swallows_no_effect_runtime_error(monkeypatch, caplog):
    """HA bare toggle-on against a virtual with no configured effect must
    not propagate RuntimeError out of the paho-mqtt callback thread.

    Regression guard for the issue surfaced after the initial fix attempt:
    `virtual.active = True` raises when there is no `_active_effect`, and
    an unhandled exception inside `on_message` kills the paho-mqtt thread.
    """
    import logging

    virtual = _PropertyVirtual(has_effect=False)
    stub = _build_handler_stub(virtual, monkeypatch)
    msg = _build_msg(virtual.id, {"state": "on"})

    with caplog.at_level(
        logging.WARNING, logger="ledfx.integrations.mqtt_hass"
    ):
        MQTT_HASS.on_message(stub, client=MagicMock(), userdata=None, msg=msg)

    assert virtual.active is False
    assert any("Cannot activate" in rec.getMessage() for rec in caplog.records)


def test_state_off_safe_on_virtual_with_no_effect(monkeypatch):
    """Toggling off a virtual that has no effect must succeed silently.

    Deactivation has no effect-required guard in `Virtual.deactivate`, so
    this path should never raise regardless of effect state.
    """
    virtual = _PropertyVirtual(has_effect=False)
    virtual._active = True
    stub = _build_handler_stub(virtual, monkeypatch)
    msg = _build_msg(virtual.id, {"state": "off"})

    MQTT_HASS.on_message(stub, client=MagicMock(), userdata=None, msg=msg)

    assert virtual.active is False
