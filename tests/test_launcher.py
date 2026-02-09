import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catalog import ToolItem

try:
    from launcher import build_cmdline
    _HAS_LAUNCHER = True
except ImportError:
    _HAS_LAUNCHER = False


def _item(kind, value, args=()):
    return ToolItem(
        id="test",
        domain="ms_settings",
        title="Test",
        description="Test item",
        launch_kind=kind,
        launch_value=value,
        args=args,
    )


@unittest.skipUnless(_HAS_LAUNCHER, "launcher deps not available")
class TestBuildCmdline(unittest.TestCase):
    def test_ms_settings_uri(self):
        cmd = build_cmdline(_item("ms_settings_uri", "ms-settings:display"))
        self.assertEqual(cmd, ["cmd", "/c", "start", "", "ms-settings:display"])

    def test_control_panel_home(self):
        cmd = build_cmdline(_item("control_panel_home", ""))
        self.assertEqual(cmd, ["control.exe"])

    def test_control_panel_name(self):
        cmd = build_cmdline(_item("control_panel_name", "Microsoft.System"))
        self.assertEqual(cmd, ["control.exe", "/name", "Microsoft.System"])

    def test_control_panel_cpl(self):
        cmd = build_cmdline(_item("control_panel_cpl", "powercfg.cpl"))
        self.assertEqual(cmd, ["control.exe", "powercfg.cpl"])

    def test_msc(self):
        cmd = build_cmdline(_item("msc", "devmgmt.msc"))
        self.assertEqual(cmd, ["mmc.exe", "devmgmt.msc"])

    def test_msc_with_args(self):
        cmd = build_cmdline(_item("msc", "devmgmt.msc", ("/s",)))
        self.assertEqual(cmd, ["mmc.exe", "devmgmt.msc", "/s"])

    def test_exe(self):
        cmd = build_cmdline(_item("exe", "taskmgr.exe"))
        self.assertEqual(cmd, ["taskmgr.exe"])

    def test_exe_with_args(self):
        cmd = build_cmdline(_item("exe", "notepad.exe", ("file.txt",)))
        self.assertEqual(cmd, ["notepad.exe", "file.txt"])

    def test_unknown_kind_fallback(self):
        cmd = build_cmdline(_item("unknown_kind", "something"))
        self.assertEqual(cmd, ["cmd", "/c", "start", "", "something"])


if __name__ == "__main__":
    unittest.main()
