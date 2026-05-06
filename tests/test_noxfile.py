import tempfile
import unittest
from pathlib import Path

import noxfile


class FakeVirtualenv:
    def __init__(self, location, reused=True):
        self.location = location
        self._reused = reused


class FakeSession:
    def __init__(self, location, reused=True):
        self.virtualenv = FakeVirtualenv(location, reused)
        self.debug_messages = []

    def debug(self, message):
        self.debug_messages.append(message)


class DependencyMetadataTests(unittest.TestCase):
    def test_should_reinstall_dependencies_does_not_write_metadata_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment_yml = Path(tmp) / "environment.yml"
            environment_yml.write_text("dependencies:\n  - python\n")
            session = FakeSession(tmp)

            should_reinstall = noxfile.should_reinstall_dependencies(
                session, environment_yml=environment_yml
            )

            self.assertTrue(should_reinstall)
            self.assertFalse((Path(tmp) / "metadata.sha1").exists())

    def test_mark_dependencies_installed_records_metadata_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment_yml = Path(tmp) / "environment.yml"
            environment_yml.write_text("dependencies:\n  - python\n")
            session = FakeSession(tmp)

            noxfile.mark_dependencies_installed(
                session, environment_yml=environment_yml
            )

            self.assertFalse(
                noxfile.should_reinstall_dependencies(
                    session, environment_yml=environment_yml
                )
            )


if __name__ == "__main__":
    unittest.main()
