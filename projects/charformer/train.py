"""Script for running training with TensorBoard logging, configurable parameters, and configuration tracking."""
import sys
from typing import Any
sys.path.append('../minformer')

import jax
import jax.numpy as jnp
import os
import functools
import argparse
from tensorboardX import SummaryWriter
import data as data
import utils as charformer_utils
import minformer.model as model
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Charformer Training Script")
    parser.add_argument("--d_model", type=int, default=1024, help="Model dimension")
    parser.add_argument("--ffw_multiplier", type=int, default=4, help="FFW multiplier")
    parser.add_argument("--query_heads", type=int, default=8, help="Number of query heads")
    parser.add_argument("--key_heads", type=int, default=8, help="Number of key heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--key_dim", type=int, default=128, help="Key dimension")
    parser.add_argument("--vocab_size", type=int, default=256, help="Vocabulary size")
    parser.add_argument("--max_seq_len", type=int, default=8192, help="Maximum sequence length")
    parser.add_argument("--max_lr", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Number of warmup steps")
    parser.add_argument("--total_steps", type=int, default=100000, help="Total number of training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--log_every", type=int, default=50, help="Log metrics every N steps")
    parser.add_argument("--eval_every", type=int, default=100, help="Eval step every N steps")
    parser.add_argument("--data_dir", type=str, default='data/tfrecords/', help="Directory containing TFRecord files")
    parser.add_argument("--log_dir", type=str, default="/tmp/logs/charformer_training", help="Base directory for TensorBoard logs")
    parser.add_argument("--log_weight_histograms", default=False, action="store_true", help="Enable logging of weight histograms")
    parser.add_argument("--checkpoint_dir", type=str, default="/tmp/charformer_checkpoints", help="Directory for saving checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=-1, help="Save checkpoint every N steps")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from the latest checkpoint")
    return parser.parse_args()

def clean_key(key):
    cleaned = key.replace("['", "").replace("']", "")
    cleaned = cleaned.replace(".", "/")
    return cleaned

def flatten_pytree(tree):
    leaves = jax.tree_util.tree_map_with_path(
        lambda p, x: (clean_key(jax.tree_util.keystr(p)), x),
        tree
    )
    return jax.tree_util.tree_leaves(leaves, is_leaf=lambda x: isinstance(x, tuple))

def log_metrics(writer, metrics, step):
    flat_metrics = flatten_pytree(metrics)
    for (key, value) in flat_metrics:
        if isinstance(value, (int, float, jnp.number)):
            writer.add_scalar(key, value, step)
        elif isinstance(value, jnp.ndarray) and value.size == 1:
            writer.add_scalar(key, value.item(), step)

def log_weight_histograms(writer, weights, step):
    flat_weights = flatten_pytree({'weights': weights})
    for key, value in flat_weights:
        if isinstance(value, jnp.ndarray):
            writer.add_histogram(key, value, step)

def eval_batch(batch: Any, dataset: data.CharDataset, weights: model.Weights, writer: SummaryWriter, step: int, cfg: model.Config, batch_row: int = 0):
    _, internals = model.compute_loss(weights, batch['x'], batch['segment_ids'], batch['y'], cfg)
    losses = internals['per_token_loss']
    length = jnp.sum(batch['segment_ids'][batch_row] != 0) - 1
    text = dataset.detokenize(batch['y'][batch_row, :length])
    losses = losses[batch_row, :length]
    charformer_utils.visualize_token_prediction_difficulty(text, losses, save_image=True, writer=writer, step=step)

def main():
    args = parse_args()

    # Create a unique log directory name with key configuration parameters
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_name = f"d{args.d_model}_l{args.num_layers}_h{args.query_heads}_lr{args.max_lr}_{timestamp}"
    log_dir = os.path.join(args.log_dir, log_dir_name)

    # Data setup
    ds = data.CharDataset(data.CharDataset.get_default_config())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, args.data_dir)
    iter = ds.create_iterator(str(data_dir) + 'record_*.tfrecord', batch_size=args.batch_size)

    # Model configuration
    cfg = model.Config(
        d_model=args.d_model,
        ffw_multiplier=args.ffw_multiplier,
        query_heads=args.query_heads,
        key_heads=args.key_heads,
        num_layers=args.num_layers,
        key_dim=args.key_dim,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        causal=True,
        use_attn_kernel=True,
        weight_dtype=jnp.float32,
        rules=model.fsdp_rules,
        mesh=model.create_mesh(),
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
    )
    # Print the config to verify all parameters are set correctly
    print("Model Configuration:")
    for field in cfg.__dataclass_fields__:
        print(f"{field}: {getattr(cfg, field)}")

    # Checkpoint manager setup
    ckpt_manager = model.make_mgnr(path=args.checkpoint_dir)

    # Initialize or load weights and optimizer state
    if args.resume_from_checkpoint:
        print("Resuming from checkpoint...")
        weights, opt_state = model.load(ckpt_manager, cfg)
        start_step = ckpt_manager.latest_step()
    else:
        print("Initializing new weights...")
        weights = model.Weights.init(cfg, jax.random.PRNGKey(0), cfg.mesh, model.fsdp_rules)
        opt_state = model.init_adam_state(weights)
        start_step = 0
        print("Initialization done.")


    # JIT-compile the update step
    step = jax.jit(model.update_step, static_argnames='cfg')
    step = functools.partial(step, cfg=cfg)

    # Training loop
    with SummaryWriter(log_dir) as writer:
        # Log hyperparameters
        writer.add_hparams({
            'd_model': cfg.d_model,
            'num_layers': cfg.num_layers,
            'query_heads': cfg.query_heads,
            'key_heads': cfg.key_heads,
            'max_lr': cfg.max_lr,
            'min_lr': cfg.min_lr,
            'warmup_steps': cfg.warmup_steps,
            'total_steps': cfg.total_steps,
            'batch_size': args.batch_size,
            'max_seq_len': cfg.max_seq_len,
        }, {})

        # TODO(sholto): If we want identical restart from mid training we need to step the
        # iterator or get a read pointer-able dataset.
        for i in range(start_step, cfg.total_steps):
            batch = next(iter)
            batch = jax.device_put(batch, model.input_shardings(cfg.mesh, cfg.rules))
            if i % args.eval_every == 0:
                eval_batch(batch, ds, weights, writer, i, cfg)

            loss, weights, opt_state, internals = step(weights, batch['x'], batch['segment_ids'], batch['y'], opt_state, i)
            
            if i % args.log_every == 0:
                # Log loss to TensorBoard
                writer.add_scalar('loss', loss, i)
                print(f"Step {i}, Loss: {loss}")
                log_metrics(writer, internals, i)

                # Log weight histograms if enabled
                if args.log_weight_histograms:
                    log_weight_histograms(writer, weights, i)
            
            # Save checkpoint
            if i % args.checkpoint_interval == 0 and args.checkpoint_interval != -1:
                print(f"Saving checkpoint at step {i}")
                model.save(ckpt_manager, weights, opt_state, i)

            # Optional: Add code here to save checkpoints periodically

    print("Training completed. TensorBoard logs saved in:", log_dir)

if __name__ == "__main__":
    main()