"""Minimal model definition."""

import functools
import jax
import jax.numpy as jnp
from flax import struct
from collections import namedtuple
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
from jax.experimental.pallas.ops.tpu import flash_attention

def create_mesh():
    """Always 1D because only care about FSDP."""
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, ('x',))
    return mesh

ShardingRules = namedtuple('FSDPRules', ['batch', 'sequence', 'd_model', 'query_heads', 'key_heads', 'key_dim', 'ffw', 'vocab'])

fsdp_rules = ShardingRules(
    batch='x',
    sequence=None,
    d_model='x',
    query_heads=None,
    key_heads=None,
    key_dim=None,
    ffw=None,
    vocab=None
)

def _logical_to_physical(logical: P, rules: ShardingRules):
    """Converts logical to physical pspec."""
    return P(*(getattr(rules, l) for l in logical))
             
def _logical_to_sharding(logical: P, mesh: jax.sharding.Mesh, rules: ShardingRules):
    """Converts logical to sharding."""
    return jax.sharding.NamedSharding(mesh, _logical_to_physical(logical, rules))

@struct.dataclass
class Config:
    d_model: int
    ffw_multiplier: int
    query_heads: int
    key_heads: int
    num_layers: int
    key_dim: int
    vocab_size: int
    causal: bool
    use_attn_kernel: bool
    # Sharding rules
    rules: ShardingRules
    mesh: jax.sharding.Mesh | None

@struct.dataclass
class Layer:
    q: jax.Array
    k: jax.Array
    v: jax.Array
    proj: jax.Array
    w1: jax.Array
    w2: jax.Array
    gamma1: jax.Array
    gamma2: jax.Array

    @classmethod
    def shape(cls, cfg: Config):
        return Layer(
            q=jax.ShapeDtypeStruct((cfg.d_model, cfg.query_heads, cfg.key_dim), jnp.bfloat16),
            k=jax.ShapeDtypeStruct((cfg.d_model, cfg.key_heads, cfg.key_dim), jnp.bfloat16),
            v=jax.ShapeDtypeStruct((cfg.d_model, cfg.key_heads, cfg.key_dim), jnp.bfloat16),
            proj=jax.ShapeDtypeStruct((cfg.query_heads, cfg.key_dim, cfg.d_model), jnp.bfloat16),
            w1=jax.ShapeDtypeStruct((cfg.d_model, cfg.d_model * cfg.ffw_multiplier), jnp.bfloat16),
            w2=jax.ShapeDtypeStruct((cfg.d_model * cfg.ffw_multiplier, cfg.d_model), jnp.bfloat16),
            gamma1=jax.ShapeDtypeStruct((cfg.d_model,), jnp.bfloat16),
            gamma2=jax.ShapeDtypeStruct((cfg.d_model,), jnp.bfloat16),
        )
    
    
    @classmethod
    def logical_axes(cls, cfg: Config):
        del cfg
        return Layer(
            q=P('d_model', 'query_heads', 'key_dim'),
            k=P('d_model', 'key_heads', 'key_dim'),
            v=P('d_model', 'key_heads', 'key_dim'),
            proj=P('query_heads', 'key_dim', 'd_model'),
            w1=P('d_model', 'ffw'),
            w2=P('ffw', 'd_model'),
            gamma1=P('d_model'),
            gamma2=P('d_model'),
        )

    @classmethod
    def shardings(cls, cfg: Config, mesh: jax.sharding.Mesh, rules: ShardingRules):
        return jax.tree.map(
            lambda logical: _logical_to_sharding(logical, mesh, rules),
            cls.logical_axes(cfg),
        )



    @classmethod
    def init(cls, cfg: Config, key: jax.random.PRNGKey):
        shape = cls.shape(cfg)
        key, *subkeys = jax.random.split(key, 7)

        return Layer(
            q=jax.random.normal(subkeys[0], shape.q.shape, shape.q.dtype) / (cfg.d_model ** 0.5),
            k=jax.random.normal(subkeys[1], shape.k.shape, shape.k.dtype) / (cfg.d_model ** 0.5),
            v=jax.random.normal(subkeys[2], shape.v.shape, shape.v.dtype) / (cfg.d_model ** 0.5),
            proj=jax.random.normal(subkeys[3], shape.proj.shape, shape.proj.dtype) / ((cfg.query_heads * cfg.key_dim) ** 0.5),
            w1=jax.random.normal(subkeys[4], shape.w1.shape, shape.w1.dtype) / (cfg.d_model ** 0.5),
            w2=jax.random.normal(subkeys[5], shape.w2.shape, shape.w2.dtype) / ((cfg.d_model * cfg.ffw_multiplier) ** 0.5),
            gamma1=jnp.ones(shape.gamma1.shape, shape.gamma1.dtype),
            gamma2=jnp.ones(shape.gamma2.shape, shape.gamma2.dtype),
        )

@struct.dataclass
class Weights:
    layers: list[Layer]
    embedding: jax.Array  # New field for token embeddings
    vocab_proj: jax.Array  # New field for final vocabulary projection


    @classmethod
    def shape(cls, cfg: Config):
        return Weights(layers=[Layer.shape(cfg) for _ in range(cfg.num_layers)],
            embedding=jax.ShapeDtypeStruct((cfg.vocab_size, cfg.d_model), jnp.bfloat16),
            vocab_proj=jax.ShapeDtypeStruct((cfg.d_model, cfg.vocab_size), jnp.bfloat16))

    @classmethod
    def logical_axes(cls, cfg: Config):
        return Weights(
            layers=[Layer.logical_axes(cfg) for _ in range(cfg.num_layers)],
            embedding=P('vocab', 'd_model'),
            vocab_proj=P('d_model', 'vocab')
        )

    @classmethod
    def shardings(cls, cfg: Config, mesh: jax.sharding.Mesh, rules: ShardingRules):
        logical_axes = cls.logical_axes(cfg)
        return Weights(
            layers=[Layer.shardings(cfg, mesh, rules) for _ in range(cfg.num_layers)],
            embedding=_logical_to_sharding(logical_axes.embedding, mesh, rules),
            vocab_proj=_logical_to_sharding(logical_axes.vocab_proj, mesh, rules),
        )

    @classmethod
    def init(cls, cfg: Config, key: jax.random.PRNGKey, mesh: jax.sharding.Mesh, rules: ShardingRules):
        # TODO(sholto): we're repeating the map here.
        # TODO(sholto): This won't get cached? I suppose we only call it once.
        @functools.partial(jax.jit, out_shardings=cls.shardings(cfg, mesh, rules))
        def _init():
            shape = cls.shape(cfg)
            keys = jax.random.split(key, cfg.num_layers + 2)  # +2 for embedding and vocab_proj
            return Weights(layers=[Layer.init(cfg, keys[l]) for l in range(cfg.num_layers)],
                embedding=jax.random.normal(keys[-2], shape.embedding.shape, shape.embedding.dtype) / (cfg.d_model ** 0.5),
                vocab_proj=jax.random.normal(keys[-1], shape.vocab_proj.shape, shape.vocab_proj.dtype) / (cfg.d_model ** 0.5)
            )
        return _init()



def _generate_fixed_pos_embedding(
    features, length, min_timescale=1.0, max_timescale=10000.0
):
  """Generate Sin/Cos for Rotary Embeddings.

  Generates sinusoids at (features//2) different timescales, where the
  timescales form a geometric series from min_timescale to max_timescale
  (max_timescale is not included, but would be the next element in the series).

  Sinusoids are evaluated at integer positions i in [0, length).

  The outputs are computed as:

    output_sin[i, j] = sin(i / timescale[j])
    output_cos[i, j] = cos(i / timescale[j])

  Args:
    features: an integer
    length: an integer
    min_timescale: an optional float
    max_timescale: an optional float

  Returns:
    output_sin: a float32 Tensor with shape [length, features // 2]
    output_cos: a float32 Tensor with shape [length, features // 2]
  """
  # Forked from
  # flaxformer/components/embedding.py;l=592
  fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
  timescale = min_timescale * (max_timescale / min_timescale) ** fraction
  rotational_frequency = 1.0 / timescale
  # Must use high precision einsum here, since rounding off to a bfloat16 is
  # catastrophic. bfloat16 rounds 257 to 256, but sin(257) is very different
  # from sin(256).
  sinusoid_inp = jnp.einsum(
      'i , j -> i j',
      jnp.arange(length),
      rotational_frequency,
      precision=jax.lax.Precision.HIGHEST,
  )
  return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)

def apply_rotary_embedding(x, sin, cos):
    assert x.ndim == 4
    x1, x2 = jnp.split(x, 2, axis=-1)
    sin, cos = sin[None, None, :, :], cos[None, None, :, :] # [T, head_dim] -> [B, h, T, head_dim]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def attention(q, k, v, segment_ids, cfg: Config):
    # TODO(sholto): Stabilise with -max.
    seq_len = q.shape[1]
    # Div sqrt(key_dim)
    scale = (q.shape[-1] ** -0.5)
    qk = jnp.einsum('bhtd,bhTd->bhtT', q, k) * scale
    # [B, t, T]
    segment_mask = segment_ids[:, None, :] == segment_ids[:, :, None]
    # [B, t, T] -> [B, 1, t, T]
    segment_mask = segment_mask[:, None, :, :]
    if cfg.causal:
        # [1, 1, t, T]
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, None, :, :]
        combined_mask = jnp.logical_and(segment_mask, causal_mask)
    else:
        combined_mask = segment_mask
    # Apply the combined mask
    qk = jnp.where(combined_mask, qk, -1e9)
    attn = jax.nn.softmax(qk, axis=-1)
    return jnp.einsum('bhtT,bhTd->bhtd', attn, v)

def attention_kernel(q, k, v, q_segment_ids, kv_segment_ids, cfg: Config):
    """Flash attention kernel!"""

    # On TPUv3, pallas seems to only work with float32.
    q, k, v = jnp.float32(q), jnp.float32(k), jnp.float32(v)
    @functools.partial(shard_map,
              mesh=cfg.mesh,
               in_specs=(
                   _logical_to_physical(P('batch', 'query_heads', 'sequence', 'key_dim'), cfg.rules),
                   _logical_to_physical(P('batch', 'key_heads', 'sequence',  'key_dim'), cfg.rules),
                   _logical_to_physical(P('batch', 'key_heads', 'sequence', 'key_dim'), cfg.rules),
                   _logical_to_physical(P('batch', 'sequence'), cfg.rules),
                   _logical_to_physical(P('batch', 'sequence'), cfg.rules),
               ),
               out_specs=_logical_to_physical(P('batch','query_heads', 'sequence', 'key_dim'), cfg.rules),
               check_rep=False)
    def _f(q, k, v, q_segment_ids, kv_segment_ids):
        print(q.shape)
        print(q_segment_ids.shape)
        segment_ids = flash_attention.SegmentIds(q_segment_ids, kv_segment_ids)
        return flash_attention.flash_attention(q, k, v, segment_ids=segment_ids, causal=True)
    return _f(q, k, v, q_segment_ids, kv_segment_ids)

def rms_norm(x, gamma):
    rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + 1e-6)
    return gamma * x / rms

def forward_layer(x, segment_ids, layer, sin, cos, cfg: Config):
    # First RMSNorm (Pre-LN for attention)
    attn_in = rms_norm(x, layer.gamma1)
    
    # Multi-head attention
    q = jnp.einsum('btd,dhq->bhtq', attn_in, layer.q)
    k = jnp.einsum('btd,dhk->bhtk', attn_in, layer.k)
    v = jnp.einsum('btd,dhv->bhtv', attn_in, layer.v)
    
    # Apply rotary embeddings
    q = apply_rotary_embedding(q, sin, cos)
    k = apply_rotary_embedding(k, sin, cos)
    
    # Compute attention
    # TODO(sholto): seperate q/kv segment ids.
    if cfg.use_attn_kernel:
        attn_out = attention_kernel(q, k, v, segment_ids, segment_ids, cfg)
    else:
        attn_out = attention(q, k, v, segment_ids, cfg)
    
    # Project attention output
    attn_out = jnp.einsum('bhtq,hqd->btd', attn_out, layer.proj)
    
    # Residual connection
    x = x + attn_out
    
    # Second RMSNorm (Pre-LN for FFN)
    ff_in = rms_norm(x, layer.gamma2)
    
    # FFN
    ff_out = jnp.einsum('btd,df->btf', ff_in, layer.w1)
    ff_out = jax.nn.gelu(ff_out)
    ff_out = jnp.einsum('btf,fd->btd', ff_out, layer.w2)
    
    # Residual connection
    x = x + ff_out
    
    return x

def forward(x, segment_ids, weights: Weights, cfg: Config):

    # Embed input tokens
    x = jnp.take(weights.embedding, x, axis=0)
    seq_len = x.shape[1]
    sin, cos = _generate_fixed_pos_embedding(cfg.key_dim, seq_len)
    
    for layer in weights.layers:
        x = forward_layer(x, segment_ids, layer, sin, cos, cfg)
    
    # Project to vocabulary size
    logits = jnp.einsum('btd,dv->btv', x, weights.vocab_proj)
    return logits

def cross_entropy_loss(logits, labels):
    num_classes = logits.shape[-1]
    labels_one_hot = jax.nn.one_hot(labels, num_classes)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(labels_one_hot * log_probs, axis=-1).mean()

def compute_loss(weights, x, segment_ids, y, cfg):
    logits = forward(x, segment_ids, weights, cfg)
    loss = cross_entropy_loss(logits, y)
    return loss

def compute_loss_and_grads(weights, x, segment_ids, y, cfg):
    return jax.value_and_grad(compute_loss)(weights, x, segment_ids, y, cfg)

def update_weights(weights, grads, lr=3e-4):
    return jax.tree.map(lambda p, g: p - g * lr, weights, grads)

def update_step(weights, x, segment_ids, y, cfg):
    loss, grads = compute_loss_and_grads(weights, x, segment_ids, y, cfg)
    weights = update_weights(weights, grads)
    return loss, weights