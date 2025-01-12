import pytest
from scipy._lib._array_api import array_namespace, is_jax, xp_assert_equal
from scipy.conftest import lazy_jax_jit


def jittable(x):
    """A jittable function"""
    return x * 2.0


def non_jittable(x):
    """This function materializes the input array, so it will fail
    when wrapped in jax.jit
    """
    xp = array_namespace(x)
    if xp.any(x < 0.0):
        raise ValueError("Negative values not allowed")
    return x


def non_jittable2(x):
    return non_jittable(x)


def static_params(x, n, flag=False):
    """Function with static parameters that must not be jitted"""
    if flag and n > 0:  # This fails if n or flag are jitted arrays
        return x * 2.0
    else:
        return x * 3.0


lazy_jax_jit(jittable)
lazy_jax_jit(non_jittable2)
lazy_jax_jit(static_params, static_argnums=(1, 2), static_argnames=("n", "flag"))


def test_lazy_jax_jit(xp):
    x = xp.asarray([1.0, 2.0])

    xp_assert_equal(jittable(x), xp.asarray([2.0, 4.0]))

    xp_assert_equal(non_jittable(x), xp.asarray([1.0, 2.0]))  # Not jitted
    if is_jax(xp):
        with pytest.raises(
            TypeError, match="Attempted boolean conversion of traced array"
        ):
            non_jittable2(x)  # Jitted
    else:
        xp_assert_equal(non_jittable2(x), xp.asarray([1.0, 2.0]))

    xp_assert_equal(static_params(x, 1), xp.asarray([3.0, 6.0]))
    xp_assert_equal(static_params(x, 1, True), xp.asarray([2.0, 4.0]))
    xp_assert_equal(static_params(x, 1, False), xp.asarray([3.0, 6.0]))
    xp_assert_equal(static_params(x, 0, False), xp.asarray([3.0, 6.0]))
    xp_assert_equal(static_params(x, 1, flag=True), xp.asarray([2.0, 4.0]))
    xp_assert_equal(static_params(x, n=1, flag=True), xp.asarray([2.0, 4.0]))
