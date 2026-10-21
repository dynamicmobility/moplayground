import subprocess
import sys

import pytest

import moplayground as mop

# Bruce / locomotion envs are out of scope for the suite (heavy MuJoCo models
# loaded at import time). The package itself is imported eagerly; its modules
# must not be.
LOCOMOTION = 'moplayground.envs.locomotion.'


def _target_module(name):
    target = mop._EXPORTS[name]
    return target[0] if isinstance(target, tuple) else target


EXPORTS = sorted(n for n in mop._EXPORTS if not _target_module(n).startswith(LOCOMOTION))


@pytest.mark.parametrize('name', EXPORTS)
def test_export_resolves(name):
    assert getattr(mop, name) is not None


def test_all_matches_exports():
    assert set(mop.__all__) == set(mop._EXPORTS) | set(mop._SUBPACKAGES)


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        mop.definitely_not_a_real_name


def test_import_is_lazy_about_locomotion():
    code = (
        'import sys, moplayground; '
        f'print(any(m.startswith({LOCOMOTION!r}) for m in sys.modules))'
    )
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == 'False'
