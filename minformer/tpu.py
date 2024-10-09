from collections import namedtuple

import jax
from jax.sharding import PartitionSpec as P


def create_mesh():
    """Always 1D because only care about FSDP."""
    devices = jax.devices()
    # Create a 1D mesh with all devices along the 'x' axis
    mesh = jax.sharding.Mesh(devices, ("x",))
    return mesh


ShardingRules = namedtuple(
    "FSDPRules",
    [
        "batch",
        "sequence",
        "d_model",
        "query_heads",
        "key_heads",
        "key_dim",
        "ffw",
        "vocab",
    ],
)

# Define sharding rules for Fully Sharded Data Parallelism (FSDP)
fsdp_rules = ShardingRules(
    batch="x",  # Shard batch dimension
    sequence=None,  # Don't shard sequence dimension
    d_model="x",  # Shard model dimension
    query_heads=None,
    key_heads=None,
    key_dim=None,
    ffw=None,
    vocab=None,
)

# Define sharding rules for model parallelism
mdl_parallel_rules = ShardingRules(
    batch=None,
    sequence=None,
    d_model=None,
    query_heads="x",  # Shard query heads
    key_heads="x",  # Shard key heads
    key_dim=None,
    ffw="x",  # Shard feed-forward layer
    vocab=None,
)


def _logical_to_physical(logical: P, rules: ShardingRules):
    """Converts logical to physical pspec."""
    return P(*(getattr(rules, axis) for axis in logical))


def _logical_to_sharding(logical: P, mesh: jax.sharding.Mesh, rules: ShardingRules):
    """Converts logical to sharding."""
    return jax.sharding.NamedSharding(mesh, _logical_to_physical(logical, rules))
