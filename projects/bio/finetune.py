"""


For open genome:

python3 projects/bio/finetune.py --pretrained_checkpoint_dir=gs://minformer_data/pretrained_ckpt/v1 --finetuned_checkpoint_dir=gs://minformer_data/finetuned_ckpt/v1 --checkpoint_interval=1000 --max_seq_len=8192 --dataset=shae_8k --log_every=10

"""

import argparse
import functools
import os
from datetime import datetime
from typing import Any

# Assuming these are in the same directory or in the Python path
import data
import data_hf
import data_shae
import jax
import jax.numpy as jnp
import modelling.model as model
import numpy as np
from jax.profiler import trace
from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser(description="DNA Sequence Training Script")
    parser.add_argument("--d_model", type=int, default=2048, help="Model dimension")
    parser.add_argument("--ffw_multiplier", type=int, default=4, help="FFW multiplier")
    parser.add_argument("--query_heads", type=int, default=8, help="Number of query heads")
    parser.add_argument("--key_heads", type=int, default=8, help="Number of key heads")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--key_dim", type=int, default=128, help="Key dimension")
    parser.add_argument("--vocab_size", type=int, default=8, help="Vocabulary size")
    parser.add_argument("--max_seq_len", type=int, default=16384, help="Maximum sequence length")
    parser.add_argument("--max_lr", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Number of warmup steps")
    parser.add_argument("--total_steps", type=int, default=30000, help="Total number of training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--log_every", type=int, default=50, help="Log metrics every N steps")
    parser.add_argument("--eval_every", type=int, default=1000, help="Evaluate model every N steps")
    parser.add_argument("--data_dir", type=str, default="data/tfrecords/", help="Directory containing TFRecord files")
    parser.add_argument("--log_dir", type=str, default="/tmp/logs/shae", help="Base directory for TensorBoard logs")
    parser.add_argument(
        "--pretrained_checkpoint_dir", type=str, default="gs://minformer_data/pretrained_ckpt/v1", help="Directory for loading checkpoints"
    )
    parser.add_argument(
        "--finetuned_checkpoint_dir", type=str, default="gs://minformer_data/finetuned_ckpt/v1", help="Directory for saving checkpoints"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["human-genome-8192", "open-genome-imgpr", "shae_8k"],
        default="open-genome-imgpr",
        help="Type of dataset to download and process",
    )
    return parser.parse_args()


def clean_key(key):
    cleaned = key.replace("['", "").replace("']", "")
    cleaned = cleaned.replace(".", "/")
    return cleaned


def flatten_pytree(tree):
    leaves = jax.tree_util.tree_map_with_path(lambda p, x: (clean_key(jax.tree_util.keystr(p)), x), tree)
    return jax.tree_util.tree_leaves(leaves, is_leaf=lambda x: isinstance(x, tuple))


def log_metrics(writer, metrics, step):
    flat_metrics = flatten_pytree(metrics)
    for key, value in flat_metrics:
        if isinstance(value, (int, float, jnp.number)):
            writer.add_scalar(key, value, step)
        elif isinstance(value, jnp.ndarray) and value.size == 1:
            writer.add_scalar(key, value.item(), step)


def compute_loss(
    weights: model.Weights,
    x: jax.Array,
    segment_ids: jax.Array,
    y: jax.Array,
    cfg: model.Config,
    aux: Any | None = None,
) -> tuple[jax.Array, Any]:
    logits, internals, x = model.forward(x, segment_ids, weights, cfg, aux=aux)
    # Important assumption that segment_ids 0 is 'padding'.
    loss_mask = jnp.where(segment_ids == 0, 0, 1)
    if not cfg.causal:
        assert "bert_mask" in aux
        # Only count the loss for tokens that were masked in BERT.
        loss_mask &= aux["bert_mask"]

    loss, accuracy, internals = model.cross_entropy_loss(logits, y, loss_mask, internals)
    internals["token_prediction_loss"] = loss
    # TODO(sholto):
    # CE for lad/sad_pred. MSE for lad/sad_regress.
    if aux is not None:
        aux_mask = jnp.ones_like(aux["lad_category"][:, 0])  # [B]
        lad_ce, lad_acc = model.cross_entropy_loss(
            internals["lad_pred"],
            aux["lad_category"][:, 0],
            aux_mask,
        )
        sad_ce, sad_acc = model.cross_entropy_loss(
            internals["sad_pred"],
            aux["sad_category"][:, 0],
            aux_mask,
        )
        lad_mse = ((internals["lad_reg"] - aux["lad_value"][:, 0]) ** 2).sum()
        sad_mse = ((internals["sad_reg"] - aux["sad_value"][:, 0]) ** 2).sum()
        (
            internals["lad_ce"],
            internals["lad_acc"],
            internals["sad_ce"],
            internals["sad_acc"],
            internals["lad_mse"],
            internals["sad_mse"],
        ) = (lad_ce, lad_acc, sad_ce, sad_acc, lad_mse, sad_mse)

    internals["accuracy"] = accuracy

    # Only fine-tune!
    loss += 1 * (lad_ce + sad_ce) + 0.1 * (lad_mse + sad_mse)
    return loss, internals


def main():
    args = parse_args()

    # Create a unique log directory name with key configuration parameters
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir_name = f"d{args.d_model}_l{args.num_layers}_h{args.query_heads}_lr{args.max_lr}_{timestamp}"
    log_dir = os.path.join(args.log_dir, log_dir_name)

    # Data setup
    # TODO: Configure.
    stage_1 = ["gs://minformer_data/shae_8k/tfrecords/record_*.tfrecord"]
    stage_2 = []
    # iter = data_shae.create_iterator(
    #     stage_1=stage_1, stage_2=stage_2, batch_size=args.batch_size, shuffle=True
    # )
    iter = data_shae.create_iterator([f"gs://minformer_data/shae_8k/tfrecords/" + "record_*.tfrecord"], [], batch_size=32, shuffle=True)
    process_batch = model.process_batch_shae

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
        weight_dtype_at_rest=jnp.float32,
        active_weight_dtype=jnp.bfloat16,
        rules=model.fsdp_rules,
        mesh=model.create_mesh(),
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.total_steps,
        # mega_byte=True,
        # patch_size=8,
    )

    # Print the config to verify all parameters are set correctly
    print("Model Configuration:")
    for field in cfg.__dataclass_fields__:
        print(f"{field}: {getattr(cfg, field)}")

    # Checkpoint manager setup
    pretrained_ckpt_manager = model.make_mngr(path=args.pretrained_checkpoint_dir)
    finetuned_ckpt_manager = model.make_mngr(path=args.finetuned_checkpoint_dir)
    # Initialize or load weights and optimizer state
    print(f"Resuming from checkpoint {args.pretrained_checkpoint_dir}")
    print(f"Saving to checkpoint {args.finetuned_checkpoint_dir}")
    weights, opt_state = model.load(pretrained_ckpt_manager, cfg)
    start_step = 0

    # JIT-compile the update step
    step = jax.jit(model.update_step, static_argnames=["cfg", "override_compute_loss_fn"])
    step = functools.partial(step, cfg=cfg, override_compute_loss_fn=compute_loss)

    # Training loop
    with SummaryWriter(log_dir) as writer:
        # Log hyperparameters
        writer.add_hparams(
            {
                "d_model": cfg.d_model,
                "num_layers": cfg.num_layers,
                "query_heads": cfg.query_heads,
                "key_heads": cfg.key_heads,
                "max_lr": cfg.max_lr,
                "min_lr": cfg.min_lr,
                "warmup_steps": cfg.warmup_steps,
                "total_steps": cfg.total_steps,
                "batch_size": args.batch_size,
                "max_seq_len": cfg.max_seq_len,
            },
            {},
        )

        for i in range(start_step, cfg.total_steps):
            next_batch = next(iter)
            batch = process_batch(next_batch, cfg, step_idx=i)
            batch = jax.device_put(batch, model.input_shardings(cfg.mesh, cfg.rules))

            # Always profile on the first step so that we can think about optimisations.
            if i == 0:
                with trace(log_dir):
                    loss, weights, opt_state, internals = step(
                        weights,
                        batch["x"],
                        batch["segment_ids"],
                        batch["y"],
                        opt_state,
                        i,
                        aux=batch["aux"],
                    )
                    jax.block_until_ready(loss)
            else:
                loss, weights, opt_state, internals = step(
                    weights, batch["x"], batch["segment_ids"], batch["y"], opt_state, i, aux=batch["aux"],
                )

            if i % args.log_every == 0:
                # Log loss and accuracy to TensorBoard
                writer.add_scalar("loss", loss, i)
                writer.add_scalar("accuracy", internals["accuracy"], i)
                writer.add_scalar("num_tokens_per_batch", np.sum(batch["segment_ids"] != 0), i)
                print(f"Step {i}, Loss: {loss}, Accuracy: {internals['accuracy']}")
                log_metrics(writer, internals, i)

            # Save checkpoint
            if i > 0 and i % args.checkpoint_interval == 0:
                print(f"Saving checkpoint at step {i}")
                model.save(finetuned_ckpt_manager, weights, opt_state, i)

    print("Training completed. TensorBoard logs saved in:", log_dir)


if __name__ == "__main__":
    main()
