import jax.numpy as jnp
from jax import tree_util


def count_params(params):
    """Counts the number of parameters in a pytree of arrays."""
    return sum(jnp.size(x) for x in tree_util.tree_leaves(params))
