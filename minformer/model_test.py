import functools
import jax
import jax.numpy as jnp
import numpy as np
import model

def print_test_passed(test_name):
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    CHECKMARK = '\u2713'
    print(f"{GREEN}{BOLD}[{CHECKMARK}] Test passed:{RESET} {test_name}")

def assert_mask_equal(actual, expected, test_name):
    if not jnp.array_equal(actual, expected):
        print(f"\n{test_name} failed!")
        print("Actual mask:")
        print(actual)
        print("\nExpected mask:")
        print(expected)
        raise AssertionError(f"{test_name} failed")

def test_make_attention_mask():
    # Test 1: Basic causal mask
    q_len, k_len = 4, 4
    q_segment_ids = jnp.array([[1, 1, 1, 1]])
    k_segment_ids = jnp.array([[1, 1, 1, 1]])
    q_offset = jnp.array([0])
    causal = True

    mask = model.make_attention_mask(q_len, k_len, q_segment_ids, k_segment_ids, q_offset, causal)
    expected_mask = jnp.array([
        [[[True, False, False, False],
          [True, True, False, False],
          [True, True, True, False],
          [True, True, True, True]]]
    ])
    assert_mask_equal(mask, expected_mask, "Basic causal mask test")
    print_test_passed("Basic causal mask")

    # Test 2: Non-causal mask
    causal = False
    mask = model.make_attention_mask(q_len, k_len, q_segment_ids, k_segment_ids, q_offset, causal)
    expected_mask = jnp.ones((1, 1, 4, 4), dtype=bool)
    assert_mask_equal(mask, expected_mask, "Non-causal mask test")

    # Test 3: Segmented causal mask
    q_segment_ids = jnp.array([[1, 1, 2, 2]])
    k_segment_ids = jnp.array([[1, 1, 2, 2]])
    causal = True
    mask = model.make_attention_mask(q_len, k_len, q_segment_ids, k_segment_ids, q_offset, causal)
    expected_mask = jnp.array([
        [[[True, False, False, False],
          [True, True, False, False],
          [False, False, True, False],
          [False, False, True, True]]]
    ])
    assert_mask_equal(mask, expected_mask, "Segmented causal mask test")
    print_test_passed("Segmented causal mask")

    # Test 4: Causal mask with offset
    q_len, k_len = 2, 4
    q_segment_ids = jnp.array([[1, 1]])
    k_segment_ids = jnp.array([[1, 1, 1, 1]])
    q_offset = jnp.array([2])
    causal = True
    mask = model.make_attention_mask(q_len, k_len, q_segment_ids, k_segment_ids, q_offset, causal)
    expected_mask = jnp.array([
        [[[True, True, True, False],
          [True, True, True, True]]]
    ])
    assert_mask_equal(mask, expected_mask, "Causal mask with offset test")
    print_test_passed("Causal mask with offset")

    # Test 5: Multiple batches
    q_len, k_len = 3, 3
    q_segment_ids = jnp.array([[1, 1, 1], [2, 2, 2]])
    k_segment_ids = jnp.array([[1, 1, 1], [2, 2, 2]])
    q_offset = jnp.array([0, 0])
    causal = True
    mask = model.make_attention_mask(q_len, k_len, q_segment_ids, k_segment_ids, q_offset, causal)
    expected_mask = jnp.array([
        [[[True, False, False],
          [True, True, False],
          [True, True, True]]],
        [[[True, False, False],
          [True, True, False],
          [True, True, True]]]
    ])
    assert_mask_equal(mask, expected_mask, "Multiple batches test")
    print_test_passed("Mask with multiple batches")


def test_attention_impl_equivalence():
    # Set up test parameters
    batch_size = 8
    seq_len = 128
    num_heads = 4
    head_dim = 64
    d_model = num_heads * head_dim

    # Create a dummy config
    rules = model.ShardingRules(
        batch='x',
        sequence=None,
        d_model='x',
        query_heads=None,
        key_heads=None,
        key_dim=None,
        ffw=None,
        vocab=None
    )

    cfg = model.Config(
        d_model=d_model,
        ffw_multiplier=4,
        query_heads=num_heads,
        key_heads=num_heads,
        num_layers=1,
        key_dim=head_dim,
        vocab_size=1000,
        max_seq_len=1024,
        causal=True,
        use_attn_kernel=True,
        weight_dtype_at_rest=jnp.float32,
        active_weight_dtype=jnp.float32,
        rules=rules,
        mesh=model.create_mesh()
    )

    # Create dummy inputs
    key = jax.random.PRNGKey(0)
    q_key, k_key, v_key, segment_key = jax.random.split(key, 4)

    q = jax.random.normal(q_key, (batch_size, num_heads, seq_len, head_dim))
    k = jax.random.normal(k_key, (batch_size, num_heads, seq_len, head_dim))
    v = jax.random.normal(v_key, (batch_size, num_heads, seq_len, head_dim))

    segment_ids = jax.random.randint(segment_key, (batch_size, seq_len), 0, 2)
    q_offset = jnp.zeros(batch_size, dtype=jnp.int32)

    # Run both attention implementations
    output_attention = model.attention(q, k, v, segment_ids, segment_ids, q_offset, cfg)
    output_attention_kernel = model.attention_kernel(q, k, v, segment_ids, segment_ids, cfg)

    # Compare outputs
    atol = 1e-2  # Absolute tolerance
    rtol = 1e-2  # Relative tolerance

    jnp.allclose(output_attention, output_attention_kernel, atol=atol, rtol=rtol)
    print_test_passed("Kernel and manual implementation equivalence")

def test_cross_entropy_loss(print_intermediates: bool = False):
    # Test case : Mixed predictions with masking
    logits = jnp.array([[2.0, 1.0, 0.0],
                            [0.0, 2.0, 1.0],
                            [1.0, 0.0, 2.0]])
    labels = jnp.array([0, 1, 2])
    mask = jnp.array([1, 1, 0])  # Last prediction is masked

    loss, _ = model.cross_entropy_loss(logits, labels, mask)

    # Manual calculation:
    # For [2.0, 1.0, 0.0] and label 0: -log(e^2 / (e^2 + e^1 + e^0)) ≈ 0.4076
    # For [0.0, 2.0, 1.0] and label 1: -log(e^2 / (e^0 + e^2 + e^1)) ≈ 0.4076
    # Average of these two: (0.4076 + 0.4076) / 2 ≈ 0.4076
    expected_loss = 0.4076

    if print_intermediates:
        print("Test case  (Mixed predictions with masking):")
        print(f"Logits:\n{logits}")
        print(f"Labels: {labels}")
        print(f"Mask: {mask}")
        print(f"Calculated loss: {loss:.6f}")
        print(f"Expected loss: {expected_loss:.6f}")
        print(f"Test {'passed' if np.isclose(loss, expected_loss, atol=1e-4) else 'failed'}\n")
    print_test_passed("Cross Entropy loss correctness.")

def test_incremental_prefill():
    # Set up the configuration
    inference_config = model.Config(
        d_model=1024,
        ffw_multiplier=4,
        query_heads=8,
        key_heads=8,
        num_layers=4,
        key_dim=128,
        vocab_size=256,
        max_seq_len=8192,
        causal=True,
        use_attn_kernel=False,
        weight_dtype_at_rest=jnp.float32,
        active_weight_dtype=jnp.float32,
        rules=model.mdl_parallel_rules,
        mesh=model.create_mesh()
    )

    # Initialize weights and cache
    key = jax.random.PRNGKey(2)
    weights = model.Weights.init(inference_config, key, inference_config.mesh, model.mdl_parallel_rules)
    prefill_cache = model.KVCache.init(cfg=inference_config, batch_size=1, max_seq_len=2048)
    # Define test input sequences
    chunk_a = jnp.array([1, 2, 3, 4, 5, 6])
    chunk_b = jnp.array([7, 8, 9])
    chunk_c = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Prepare chunks
    chunk_a, segment_ids_a = model.prepare_chunk(chunk_a, pad_to=16, pad_id=0)
    chunk_b, segment_ids_b = model.prepare_chunk(chunk_b, pad_to=16, pad_id=0)
    chunk_c, segment_ids_c = model.prepare_chunk(chunk_c, pad_to=16, pad_id=0)

    # Run incremental prefill
    logits_c, prefill_cache_c, _ = model.forward(chunk_c, segment_ids_c, weights, inference_config, prefill_cache)

    prefill_cache = model.KVCache.init(cfg=inference_config, batch_size=1, max_seq_len=2048)
    logits_a, prefill_cache, _ = model.forward(chunk_a, segment_ids_a, weights, inference_config, prefill_cache)
    logits_b, prefill_cache, _ = model.forward(chunk_b, segment_ids_b, weights, inference_config, prefill_cache)

    # Assert cache lengths
    assert jnp.array_equal(prefill_cache_c.lengths, jnp.array([9])), "Cache length mismatch for chunk_c"
    assert jnp.array_equal(prefill_cache.lengths, jnp.array([9])), "Cache length mismatch for incremental chunks"

    # Assert logits shapes
    assert logits_c.shape == logits_a.shape == (1, 16, inference_config.vocab_size), "Logits shape mismatch"

    # Assert cache consistency
    for layer in range(inference_config.num_layers):
        np.testing.assert_allclose(prefill_cache_c.k[layer][:, :, :9, :].astype(jnp.float32), prefill_cache.k[layer][:, :, :9, :].astype(jnp.float32), rtol=2e-2)
        np.testing.assert_allclose(prefill_cache_c.v[layer][:, :, :9, :].astype(jnp.float32), prefill_cache.v[layer][:, :, :9, :].astype(jnp.float32), rtol=2e-2)

    # Assert logits consistency for the first 6 tokens
    assert jnp.array_equal(jnp.argmax(logits_c[0, :6], axis=-1), jnp.argmax(logits_a[0, :6], axis=-1)), "Logits mismatch for first 6 tokens"
    assert jnp.array_equal(jnp.argmax(logits_c[0, 6:9], axis=-1), jnp.argmax(logits_b[0, :3], axis=-1)), "Logits mismatch for tokens 6 to 9"

    print_test_passed("Incremental prefill correctness.")

def test_overtrain_and_sample_simple_sequence():
    # TODO(sholto): Extend with multiple sequence ids?
    cfg = model.Config(
        d_model=256,
        ffw_multiplier=4,
        query_heads=8,
        key_heads=8,
        num_layers=4,
        key_dim=128,
        vocab_size=256,
        max_seq_len=8192,
        causal=True,
        use_attn_kernel=True,
        weight_dtype_at_rest=jnp.float32,
        active_weight_dtype=jnp.float32,
        rules=model.fsdp_rules,
        mesh=model.create_mesh(),
        max_lr=3e-4,
        min_lr=1e-5,
        warmup_steps=10,
        total_steps=100,
    )

    inference_config = model.Config(
        d_model=256,
        ffw_multiplier=4,
        query_heads=8,
        key_heads=8,
        num_layers=4,
        key_dim=128,
        vocab_size=256,
        max_seq_len=8192,
        causal=True,
        use_attn_kernel=False,
        weight_dtype_at_rest=jnp.float32,
        active_weight_dtype=jnp.float32,
        rules=model.mdl_parallel_rules,
        mesh=model.create_mesh()
    )
    weights = model.Weights.init(cfg, jax.random.PRNGKey(0), cfg.mesh, model.fsdp_rules)
    opt_state = model.init_adam_state(weights)
    step = jax.jit(model.update_step, static_argnames='cfg')
    step = functools.partial(step, cfg=cfg)

    test_batch = jnp.arange(1, 256+2)[None, :]
    test_batch = jnp.repeat(test_batch, repeats = 8, axis=0)
    batch = {
        'x': test_batch[:, :-1],
        'y': test_batch[:, 1:],
        'segment_ids': jnp.ones((8, 256)),
    }
    batch = jax.device_put(batch, model.input_shardings(cfg.mesh, cfg.rules))

    ckpt_path = '/tmp/test_dir'
    ckpt_manager = model.make_mgnr(path=ckpt_path, erase=True)
    
    losses = []
    for s in range(0, 50):
        loss, weights, opt_state, _ = step(weights, batch['x'], batch['segment_ids'], batch['y'], opt_state, s)
        losses.append(loss)

        if s % 25 == 0:
            model.save(ckpt_manager, weights, opt_state, s)
    
    prompt = jnp.arange(1, 60)
    cache = model.KVCache.init(cfg=inference_config, batch_size=1, max_seq_len=2048)
    tokens, cache = model.sample_from_prompt(prompt, weights, cache, inference_config, batch_idx=0, num_steps=13)
    # Validate that we do indeed sample the sequence we overtrained in.
    assert jnp.array_equal(jnp.array(tokens), jnp.arange(60, 60+13))
    print_test_passed("Overtrain and sample from simple model.")

    weights, opt_state = model.load(ckpt_manager, cfg, step=25)
    new_losses = []
    for s in range(26, 50):
        loss, weights, opt_state, _ = step(weights, batch['x'], batch['segment_ids'], batch['y'], opt_state, s)
        new_losses.append(loss)
    assert losses[26:50] == new_losses[0:24]
    print_test_passed("Reload mid-run checkpoint with identical training continuation.")


# Run the test
if __name__ == "__main__":
    test_make_attention_mask()
    test_attention_impl_equivalence()
    test_cross_entropy_loss()
    test_incremental_prefill()
    test_overtrain_and_sample_simple_sequence()
