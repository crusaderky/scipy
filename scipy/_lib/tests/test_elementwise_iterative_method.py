import pytest

from scipy._lib._array_api import xp_assert_close, xp_assert_equal
import scipy._lib._elementwise_iterative_method as eim

skip_xp_backends = pytest.mark.skip_xp_backends
xfail_xp_backends = pytest.mark.xfail_xp_backends

