import pytest

from scipy._lib._array_api import xp_assert_close, xp_assert_equal
import scipy._lib._elementwise_iterative_method as eim
from scipy.conftest import array_api_compatible

skip_xp_backends = pytest.mark.skip_xp_backends
xfail_xp_backends = pytest.mark.xfail_xp_backends
pytestmark = [array_api_compatible,
              pytest.mark.usefixtures("skip_xp_backends"),
              pytest.mark.usefixtures("xfail_xp_backends")]

