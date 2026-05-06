import tempfile
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


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
    def __init__(self, location, reused=True, run_output=None):
        self.virtualenv = FakeVirtualenv(location, reused)
        self.debug_messages = []
        self.run_output = run_output
        self.runs = []

    def debug(self, message):
        self.debug_messages.append(message)

    def run(self, *args, **kwargs):
        self.runs.append((args, kwargs))
        self.run_args = args
        self.run_kwargs = kwargs
        return self.run_output


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


class MetatomicRCOverridesTests(unittest.TestCase):
    def test_rc_conda_dependencies_force_python_313(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment_yml = Path(tmp) / "environment.yml"
            environment_yml.write_text("dependencies:\n  - python=3.12\n")
            session_dir = Path(tmp) / "session"
            session_dir.mkdir()
            session = FakeSession(str(session_dir))
            environment = {
                "dependencies": [
                    "python=3.12",
                    "numpy",
                ]
            }
            dumped = {}

            def capture_dump(data, fd):
                dumped["environment"] = data

            with (
                mock.patch.dict(
                    os.environ, {"METATOMIC_RC_CHANNEL": "file:///tmp/rc-channel"}
                ),
                mock.patch.object(noxfile.yaml, "safe_load", return_value=environment),
                mock.patch.object(
                    noxfile.yaml, "safe_dump", side_effect=capture_dump
                ),
            ):
                noxfile.apply_metatomic_rc_overrides(environment_yml, session)

        self.assertEqual(
            dumped["environment"]["dependencies"],
            [
                "python =3.13",
                "numpy",
            ],
        )

    def test_rc_pip_dependencies_move_metatomic_stack_to_conda(self):
        pip_dependencies = [
            "--extra-index-url https://download.pytorch.org/whl/cpu",
            "torch==2.9.1",
            "metatensor-torch>=0.8,<0.9",
            "metatensor-operations",
            "metatomic-torch==0.1.7",
            "metatrain",
            "ase",
            "featomic-torch",
            "torch-pme>=0.3.1,<0.4",
        ]

        self.assertEqual(
            noxfile._rc_conda_dependencies_from_pip(pip_dependencies),
            [
                "pytorch-cpu =2.10.*=cpu_generic*",
                "python-metatensor-torch =0.9.0.rc5",
                "python-metatensor-operations =0.5.0.rc2",
                "python-metatomic-torch =0.1.12.rc2",
                "metatrain =2026.2.1.dev47",
                "python-metatensor-core =0.2.0.rc3",
            ],
        )
        self.assertEqual(
            noxfile._rc_pip_dependencies(pip_dependencies),
            [
                "ase",
                "featomic-torch @ git+https://github.com/HaoZeke/featomic.git"
                "@rc/metatomic-0.1.12#subdirectory=python/featomic_torch",
                "torch-pme @ git+https://github.com/HaoZeke/torch-pme.git"
                "@rc/metatomic-0.1.12-v0.3.2",
            ],
        )

    def test_rc_run_env_sets_metatomic_torch_build_version(self):
        session = FakeSession("/tmp")

        with mock.patch.dict(
            os.environ, {"METATOMIC_RC_CHANNEL": "file:///tmp/rc-channel"}
        ):
            noxfile._run_with_metatomic_rc_env(session, "conda", "env", "update")

        self.assertEqual(
            session.run_kwargs["env"],
            {
                "FEATOMIC_TORCH_BUILD_WITH_TORCH_VERSION": "2.10.*",
                "METATOMIC_NO_LOCAL_DEPS": "1",
                "METATOMIC_TORCH_BUILD_WITH_TORCH_VERSION": "2.10.*",
            },
        )

    def test_rc_uninstalls_pip_packages_replaced_by_conda(self):
        session = FakeSession(
            "/tmp",
            run_output="metatensor-core\nmetatomic-torch\n",
        )

        with mock.patch.dict(
            os.environ, {"METATOMIC_RC_CHANNEL": "file:///tmp/rc-channel"}
        ):
            noxfile.uninstall_metatomic_rc_pip_packages(session)

        self.assertEqual(session.runs[0][0][:2], ("python", "-c"))
        self.assertEqual(
            session.runs[1][0],
            (
                "python",
                "-m",
                "pip",
                "uninstall",
                "--yes",
                "metatensor-core",
                "metatomic-torch",
            ),
        )


if __name__ == "__main__":
    unittest.main()
