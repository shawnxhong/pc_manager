import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Stub out heavy native dependencies that may not be installed in test environments.
for mod_name in [
    "numpy", "np", "torch", "transformers",
    "transformers.AutoTokenizer", "transformers.AutoModel",
]:
    sys.modules.setdefault(mod_name, MagicMock())

from catalog import ToolItem, normalize_catalog, build_embedding_text
from launcher import build_cmdline, _validate_launch_value


class NormalizeCatalogTests(unittest.TestCase):
    """Tests for catalog normalization logic."""

    def _make_item(self, **overrides):
        base = {
            "id": "ms-settings:display",
            "title": "Display settings",
            "description": "Adjust display settings",
            "launch": {"kind": "uri", "value": "ms-settings:display", "args": []},
        }
        base.update(overrides)
        return base

    def test_basic_normalization(self):
        items = normalize_catalog([self._make_item()], [], [])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].id, "ms-settings:display")
        self.assertEqual(items[0].domain, "ms_settings")
        self.assertEqual(items[0].launch_kind, "ms_settings_uri")

    def test_deduplication(self):
        item = self._make_item()
        items = normalize_catalog([item, item], [], [])
        self.assertEqual(len(items), 1)

    def test_missing_id_skipped(self):
        item = self._make_item(id="")
        items = normalize_catalog([item], [], [])
        self.assertEqual(len(items), 0)

    def test_missing_title_skipped(self):
        item = self._make_item(title="")
        items = normalize_catalog([item], [], [])
        self.assertEqual(len(items), 0)

    def test_invalid_kind_skipped(self):
        item = self._make_item(launch={"kind": "unknown_kind", "value": "something"})
        items = normalize_catalog([item], [], [])
        self.assertEqual(len(items), 0)

    def test_kind_alias_mapping(self):
        """Kind aliases like 'uri' should map to 'ms_settings_uri'."""
        item = self._make_item(launch={"kind": "uri", "value": "ms-settings:wifi"})
        item["id"] = "ms-settings:wifi"
        items = normalize_catalog([item], [], [])
        self.assertEqual(items[0].launch_kind, "ms_settings_uri")

    def test_ms_settings_uri_validation(self):
        """ms_settings_uri items must have value starting with 'ms-settings:'."""
        item = self._make_item(launch={"kind": "uri", "value": "bad-prefix:something"})
        items = normalize_catalog([item], [], [])
        self.assertEqual(len(items), 0)

    def test_control_panel_normalization(self):
        cp_item = {
            "id": "cp-system",
            "title": "System",
            "description": "System properties",
            "launch": {"kind": "name", "value": "Microsoft.System"},
        }
        items = normalize_catalog([], [cp_item], [])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].domain, "control_panel")
        self.assertEqual(items[0].launch_kind, "control_panel_name")

    def test_system_tool_normalization(self):
        st_item = {
            "id": "devmgmt",
            "title": "Device Manager",
            "description": "Manage hardware devices",
            "launch": {"kind": "msc", "value": "devmgmt.msc"},
        }
        items = normalize_catalog([], [], [st_item])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].domain, "system_tool")
        self.assertEqual(items[0].launch_kind, "msc")

    def test_exe_empty_value_skipped(self):
        item = {
            "id": "bad-exe",
            "title": "Bad",
            "description": "No value",
            "launch": {"kind": "exe", "value": ""},
        }
        items = normalize_catalog([], [], [item])
        self.assertEqual(len(items), 0)

    def test_dict_input_format(self):
        """normalize_catalog should handle dict[id->dict] input format."""
        data = {
            "ms-settings:display": {
                "title": "Display",
                "description": "Display settings",
                "launch": {"kind": "uri", "value": "ms-settings:display"},
            }
        }
        items = normalize_catalog(data, [], [])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].id, "ms-settings:display")

    def test_none_input(self):
        items = normalize_catalog(None, None, None)
        self.assertEqual(len(items), 0)


class BuildEmbeddingTextTests(unittest.TestCase):
    def test_contains_key_fields(self):
        item = ToolItem(
            id="ms-settings:display",
            domain="ms_settings",
            title="Display settings",
            description="Adjust brightness and resolution",
            launch_kind="ms_settings_uri",
            launch_value="ms-settings:display",
        )
        text = build_embedding_text(item)
        self.assertIn("ms_settings", text)
        self.assertIn("Display settings", text)
        self.assertIn("ms-settings:display", text)
        self.assertIn("LAUNCH_KIND=ms_settings_uri", text)


class BuildCmdlineTests(unittest.TestCase):
    """Tests for build_cmdline covering each launch kind."""

    def test_ms_settings_uri(self):
        item = ToolItem(
            id="t", domain="ms_settings", title="t", description="d",
            launch_kind="ms_settings_uri", launch_value="ms-settings:display",
        )
        cmd = build_cmdline(item)
        self.assertEqual(cmd, ["cmd", "/c", "start", "", "ms-settings:display"])

    def test_control_panel_home(self):
        item = ToolItem(
            id="t", domain="control_panel", title="t", description="d",
            launch_kind="control_panel_home", launch_value="",
        )
        cmd = build_cmdline(item)
        self.assertEqual(cmd, ["control.exe"])

    def test_control_panel_name(self):
        item = ToolItem(
            id="t", domain="control_panel", title="t", description="d",
            launch_kind="control_panel_name", launch_value="Microsoft.System",
        )
        cmd = build_cmdline(item)
        self.assertEqual(cmd, ["control.exe", "/name", "Microsoft.System"])

    def test_control_panel_cpl(self):
        item = ToolItem(
            id="t", domain="control_panel", title="t", description="d",
            launch_kind="control_panel_cpl", launch_value="powercfg.cpl",
        )
        cmd = build_cmdline(item)
        self.assertEqual(cmd, ["control.exe", "powercfg.cpl"])

    def test_msc(self):
        item = ToolItem(
            id="t", domain="system_tool", title="t", description="d",
            launch_kind="msc", launch_value="devmgmt.msc",
        )
        cmd = build_cmdline(item)
        self.assertEqual(cmd, ["mmc.exe", "devmgmt.msc"])

    def test_msc_with_args(self):
        item = ToolItem(
            id="t", domain="system_tool", title="t", description="d",
            launch_kind="msc", launch_value="devmgmt.msc", args=("/s",),
        )
        cmd = build_cmdline(item)
        self.assertEqual(cmd, ["mmc.exe", "devmgmt.msc", "/s"])

    def test_exe(self):
        item = ToolItem(
            id="t", domain="system_tool", title="t", description="d",
            launch_kind="exe", launch_value="taskmgr.exe",
        )
        cmd = build_cmdline(item)
        self.assertEqual(cmd, ["taskmgr.exe"])

    def test_exe_with_args(self):
        item = ToolItem(
            id="t", domain="system_tool", title="t", description="d",
            launch_kind="exe", launch_value="notepad.exe", args=("file.txt",),
        )
        cmd = build_cmdline(item)
        self.assertEqual(cmd, ["notepad.exe", "file.txt"])


class ValidateLaunchValueTests(unittest.TestCase):
    """Tests for _validate_launch_value security checks."""

    def _item(self, kind, value, args=()):
        return ToolItem(
            id="t", domain="ms_settings", title="t", description="d",
            launch_kind=kind, launch_value=value, args=args,
        )

    def test_valid_ms_settings(self):
        self.assertIsNone(_validate_launch_value(self._item("ms_settings_uri", "ms-settings:display")))

    def test_invalid_ms_settings_prefix(self):
        err = _validate_launch_value(self._item("ms_settings_uri", "http://evil.com"))
        self.assertIsNotNone(err)

    def test_valid_cpl(self):
        self.assertIsNone(_validate_launch_value(self._item("control_panel_cpl", "powercfg.cpl")))

    def test_invalid_cpl_extension(self):
        err = _validate_launch_value(self._item("control_panel_cpl", "evil.exe"))
        self.assertIsNotNone(err)

    def test_valid_msc(self):
        self.assertIsNone(_validate_launch_value(self._item("msc", "devmgmt.msc")))

    def test_invalid_msc_extension(self):
        err = _validate_launch_value(self._item("msc", "devmgmt.exe"))
        self.assertIsNotNone(err)

    def test_valid_exe(self):
        self.assertIsNone(_validate_launch_value(self._item("exe", "taskmgr.exe")))

    def test_invalid_exe_extension(self):
        err = _validate_launch_value(self._item("exe", "taskmgr.msc"))
        self.assertIsNotNone(err)

    def test_shell_metacharacters_rejected(self):
        err = _validate_launch_value(self._item("exe", "calc.exe; rm -rf /"))
        self.assertIsNotNone(err)
        self.assertIn("metacharacters", err)

    def test_shell_metacharacters_in_args(self):
        err = _validate_launch_value(self._item("exe", "notepad.exe", args=("$(whoami)",)))
        self.assertIsNotNone(err)

    def test_pipe_rejected(self):
        err = _validate_launch_value(self._item("exe", "cmd.exe | evil"))
        self.assertIsNotNone(err)

    def test_backtick_rejected(self):
        err = _validate_launch_value(self._item("exe", "calc.exe`id`"))
        self.assertIsNotNone(err)


if __name__ == "__main__":
    unittest.main()
