import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import ToolItem, normalize_catalog, build_embedding_text


def _ms_item(**overrides):
    base = {
        "id": "ms-settings:display",
        "title": "Display",
        "description": "Display settings",
        "launch": {"kind": "uri", "value": "ms-settings:display", "args": []},
    }
    base.update(overrides)
    return base


class TestNormalizeCatalog(unittest.TestCase):
    def test_valid_ms_settings_item(self):
        result = normalize_catalog([_ms_item()], [], [])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "ms-settings:display")
        self.assertEqual(result[0].domain, "ms_settings")

    def test_valid_control_panel_item(self):
        item = {
            "id": "cpanel-system",
            "title": "System",
            "description": "System control panel",
            "launch": {"kind": "name", "value": "Microsoft.System", "args": []},
        }
        result = normalize_catalog([], [item], [])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].domain, "control_panel")
        self.assertEqual(result[0].launch_kind, "control_panel_name")

    def test_valid_system_tool_item(self):
        item = {
            "id": "taskmgr",
            "title": "Task Manager",
            "description": "Windows Task Manager",
            "launch": {"kind": "exe", "value": "taskmgr.exe", "args": []},
        }
        result = normalize_catalog([], [], [item])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].domain, "system_tool")
        self.assertEqual(result[0].launch_kind, "exe")

    def test_missing_title_rejected(self):
        result = normalize_catalog([_ms_item(title="")], [], [])
        self.assertEqual(len(result), 0)

    def test_missing_id_rejected(self):
        result = normalize_catalog([_ms_item(id="")], [], [])
        self.assertEqual(len(result), 0)

    def test_missing_description_rejected(self):
        result = normalize_catalog([_ms_item(description="")], [], [])
        self.assertEqual(len(result), 0)

    def test_invalid_launch_kind_rejected(self):
        item = _ms_item()
        item["launch"]["kind"] = "invalid_kind"
        result = normalize_catalog([item], [], [])
        self.assertEqual(len(result), 0)

    def test_ms_settings_uri_must_start_with_prefix(self):
        item = _ms_item()
        item["launch"]["value"] = "not-ms-settings"
        result = normalize_catalog([item], [], [])
        self.assertEqual(len(result), 0)

    def test_empty_exe_value_rejected(self):
        item = {
            "id": "bad-exe",
            "title": "Bad",
            "description": "Bad exe",
            "launch": {"kind": "exe", "value": "", "args": []},
        }
        result = normalize_catalog([], [], [item])
        self.assertEqual(len(result), 0)

    def test_deduplication(self):
        item = _ms_item()
        result = normalize_catalog([item, item], [], [])
        self.assertEqual(len(result), 1)

    def test_kind_alias_uri(self):
        result = normalize_catalog([_ms_item()], [], [])
        self.assertEqual(result[0].launch_kind, "ms_settings_uri")

    def test_kind_alias_cpl(self):
        item = {
            "id": "power",
            "title": "Power",
            "description": "Power options",
            "launch": {"kind": "cpl", "value": "powercfg.cpl", "args": []},
        }
        result = normalize_catalog([], [item], [])
        self.assertEqual(result[0].launch_kind, "control_panel_cpl")

    def test_kind_alias_home(self):
        item = {
            "id": "cpanel-home",
            "title": "Control Panel",
            "description": "Control Panel Home",
            "launch": {"kind": "home", "value": "", "args": []},
        }
        result = normalize_catalog([], [item], [])
        self.assertEqual(result[0].launch_kind, "control_panel_home")

    def test_dict_input_format(self):
        items = {
            "ms-settings:display": {
                "title": "Display",
                "description": "Display settings",
                "launch": {"kind": "uri", "value": "ms-settings:display", "args": []},
            }
        }
        result = normalize_catalog(items, {}, {})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, "ms-settings:display")

    def test_none_input(self):
        result = normalize_catalog(None, None, None)
        self.assertEqual(len(result), 0)

    def test_args_normalized_to_tuple(self):
        result = normalize_catalog([_ms_item()], [], [])
        self.assertIsInstance(result[0].args, tuple)

    def test_msc_kind(self):
        item = {
            "id": "devmgmt",
            "title": "Device Manager",
            "description": "Device Manager",
            "launch": {"kind": "msc", "value": "devmgmt.msc", "args": []},
        }
        result = normalize_catalog([], [], [item])
        self.assertEqual(result[0].launch_kind, "msc")


class TestBuildEmbeddingText(unittest.TestCase):
    def test_contains_all_fields(self):
        item = ToolItem(
            id="test",
            domain="ms_settings",
            title="Test Title",
            description="Test description",
            launch_kind="ms_settings_uri",
            launch_value="ms-settings:test",
        )
        text = build_embedding_text(item)
        self.assertIn("Test Title", text)
        self.assertIn("Test description", text)
        self.assertIn("ms_settings", text)
        self.assertIn("ms_settings_uri", text)
        self.assertIn("ms-settings:test", text)

    def test_returns_string(self):
        item = ToolItem(
            id="t",
            domain="system_tool",
            title="T",
            description="D",
            launch_kind="exe",
            launch_value="t.exe",
        )
        self.assertIsInstance(build_embedding_text(item), str)


class TestToolItemToDict(unittest.TestCase):
    def test_round_trip(self):
        item = ToolItem(
            id="ms-settings:display",
            domain="ms_settings",
            title="Display",
            description="Display settings",
            launch_kind="ms_settings_uri",
            launch_value="ms-settings:display",
            args=("/flag",),
        )
        d = item.to_dict()
        self.assertEqual(d["id"], "ms-settings:display")
        self.assertEqual(d["launch"]["kind"], "ms_settings_uri")
        self.assertEqual(d["launch"]["args"], ["/flag"])


if __name__ == "__main__":
    unittest.main()
