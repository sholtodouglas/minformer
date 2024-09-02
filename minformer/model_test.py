import jax
import jax.numpy as jnp
import numpy as np
import model

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

    print("All tests passed!")

def test_incremental_prefill():
    # Set up the configuration
    inference_config = model.Config(
        d_model=1024,
        ffw_multiplier=4,
        query_heads=8,
        key_heads=8,
        num_layers=8,
        key_dim=128,
        vocab_size=256,
        max_seq_len=8192,
        causal=True,
        use_attn_kernel=False,
        rules=model.mdl_parallel_rules,
        mesh=model.create_mesh()
    )

    # Initialize weights and cache
    key = jax.random.PRNGKey(0)
    weights = model.Weights.init(inference_config, key, inference_config.mesh, model.mdl_parallel_rules)
    weights = jax.tree.map(lambda x: x.astype(jnp.float32), weights)
    prefill_cache = model.KVCache.init(cfg=inference_config, batch_size=1, max_seq_len=2048)

    # Define test input sequences
    chunk_a = jnp.array([1, 2, 3, 4, 5, 6])
    chunk_b = jnp.array([7, 8, 9])
    chunk_c = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

    # Helper function to prepare chunks
    def prepare_chunk(chunk, pad_to: int):
        chunk = jnp.pad(chunk, (0, pad_to-len(chunk)))[None, :]
        segment_ids = jnp.where(chunk != 0, 1, 0).astype(jnp.int32)
        return chunk, segment_ids

    # Prepare chunks
    chunk_a, segment_ids_a = prepare_chunk(chunk_a, pad_to=16)
    chunk_b, segment_ids_b = prepare_chunk(chunk_b, pad_to=16)
    chunk_c, segment_ids_c = prepare_chunk(chunk_c, pad_to=16)

    # Run incremental prefill
    logits_c, prefill_cache_c = model.forward(chunk_c, segment_ids_c, weights, inference_config, prefill_cache)

    prefill_cache = model.KVCache.init(cfg=inference_config, batch_size=1, max_seq_len=2048)
    logits_a, prefill_cache = model.forward(chunk_a, segment_ids_a, weights, inference_config, prefill_cache)
    logits_b, prefill_cache = model.forward(chunk_b, segment_ids_b, weights, inference_config, prefill_cache)

    # Assert cache lengths
    assert jnp.array_equal(prefill_cache_c.lengths, jnp.array([9])), "Cache length mismatch for chunk_c"
    assert jnp.array_equal(prefill_cache.lengths, jnp.array([9])), "Cache length mismatch for incremental chunks"

    # Assert logits shapes
    assert logits_c.shape == logits_a.shape == (1, 16, inference_config.vocab_size), "Logits shape mismatch"

    # Assert cache consistency
    for layer in range(inference_config.num_layers):
        assert jnp.array_equal(prefill_cache_c.k[layer][:, :, :9, :], prefill_cache.k[layer][:, :, :9, :]), f"K cache mismatch at layer {layer}"
        assert jnp.array_equal(prefill_cache_c.v[layer][:, :, :9, :], prefill_cache.v[layer][:, :, :9, :]), f"V cache mismatch at layer {layer}"

    # Assert logits consistency for the first 6 tokens
    assert jnp.array_equal(jnp.argmax(logits_c[0, :6], axis=-1), jnp.argmax(logits_a[0, :6], axis=-1)), "Logits mismatch for first 6 tokens"
    assert jnp.array_equal(jnp.argmax(logits_c[0, 6:9], axis=-1), jnp.argmax(logits_b[0, :3], axis=-1)), "Logits mismatch for tokens 6 to 9"

    print("All tests passed!")

# Run the test
if __name__ == "__main__":
    test_make_attention_mask()
    test_incremental_prefill()