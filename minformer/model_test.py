import jax
import jax.numpy as jnp
import numpy as np
from model import make_attention_mask

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

    mask = make_attention_mask(q_len, k_len, q_segment_ids, k_segment_ids, q_offset, causal)
    expected_mask = jnp.array([
        [[[True, False, False, False],
          [True, True, False, False],
          [True, True, True, False],
          [True, True, True, True]]]
    ])
    assert_mask_equal(mask, expected_mask, "Basic causal mask test")

    # Test 2: Non-causal mask
    causal = False
    mask = make_attention_mask(q_len, k_len, q_segment_ids, k_segment_ids, q_offset, causal)
    expected_mask = jnp.ones((1, 1, 4, 4), dtype=bool)
    assert_mask_equal(mask, expected_mask, "Non-causal mask test")

    # Test 3: Segmented causal mask
    q_segment_ids = jnp.array([[1, 1, 2, 2]])
    k_segment_ids = jnp.array([[1, 1, 2, 2]])
    causal = True
    mask = make_attention_mask(q_len, k_len, q_segment_ids, k_segment_ids, q_offset, causal)
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
    mask = make_attention_mask(q_len, k_len, q_segment_ids, k_segment_ids, q_offset, causal)
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
    mask = make_attention_mask(q_len, k_len, q_segment_ids, k_segment_ids, q_offset, causal)
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

if __name__ == "__main__":
    test_make_attention_mask()