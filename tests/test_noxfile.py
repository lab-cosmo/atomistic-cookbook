import tempfile
import sys
import types
import unittest
from pathlib import Path


def _install_nox_stub():
    nox = types.ModuleType("nox")
    nox.options = types.SimpleNamespace()
    nox.needs_version = None

    def session(*decorator_args, **decorator_kwargs):
        if decorator_args and callable(decorator_args[0]):
            return decorator_args[0]

        def decorate(function):
            return function

        return decorate

    nox.session = session
    sys.modules["nox"] = nox


def _install_docutils_stub():
    core = types.ModuleType("docutils.core")
    core.publish_doctree = lambda *args, **kwargs: None
    core.publish_parts = lambda *args, **kwargs: {}

    nodes = types.ModuleType("docutils.nodes")
    nodes.comment = dict
    nodes.paragraph = dict
    nodes.title = dict

    rst = types.ModuleType("docutils.parsers.rst")

    class Directive:
        has_content = False

    rst.Directive = Directive
    rst.directives = types.SimpleNamespace(
        register_directive=lambda *args, **kwargs: None
    )

    sys.modules["docutils"] = types.ModuleType("docutils")
    sys.modules["docutils.core"] = core
    sys.modules["docutils.nodes"] = nodes
    sys.modules["docutils.parsers"] = types.ModuleType("docutils.parsers")
    sys.modules["docutils.parsers.rst"] = rst


def _install_yaml_stub():
    yaml = types.ModuleType("yaml")
    yaml.safe_dump = lambda *args, **kwargs: None
    yaml.safe_load = lambda *args, **kwargs: {}
    sys.modules["yaml"] = yaml


_install_nox_stub()
_install_docutils_stub()
_install_yaml_stub()

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

    def test_bare_dependency_hash_marker_forces_reinstall(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment_yml = Path(tmp) / "environment.yml"
            environment_yml.write_text("dependencies:\n  - python\n")
            session = FakeSession(tmp)
            (Path(tmp) / "metadata.sha1").write_text(
                noxfile.dependency_metadata_sha1(environment_yml=environment_yml)
            )

            self.assertTrue(
                noxfile.should_reinstall_dependencies(
                    session, environment_yml=environment_yml
                )
            )


if __name__ == "__main__":
    unittest.main()
