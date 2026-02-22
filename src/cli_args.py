import argparse


def _validate_bool_pair(args, true_flag: str, false_flag: str):
    if getattr(args, true_flag) and getattr(args, false_flag):
        raise ValueError(f"Cannot set both --{true_flag.replace('_', '-')} and --{false_flag.replace('_', '-')}")


def add_common_data_args(parser: argparse.ArgumentParser):
    parser.add_argument("--data_root", type=str, default=None, help="Override dataset root")
    parser.add_argument("--dataset_name", type=str, default=None, help="Dataset subdirectory name (e.g. ASL_gloss, ASL_sentence)")
    parser.add_argument("--batch_size", type=int, default=None, help="Train batch size")
    parser.add_argument("--val_batch_size", type=int, default=None, help="Validation batch size")


def add_vqvae_experiment_args(parser: argparse.ArgumentParser):
    add_common_data_args(parser)

    parser.add_argument("--model_max_seq_len", type=int, default=None)
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--codebook_size", type=int, default=None)
    parser.add_argument("--downsample_rate", type=int, default=None)
    parser.add_argument("--motion_encoder_layers", type=int, default=None)
    parser.add_argument("--motion_decoder_layers", type=int, default=None)
    parser.add_argument("--motion_encoder_hidden_dim", type=int, default=None)
    parser.add_argument("--motion_decoder_hidden_dim", type=int, default=None)

    parser.add_argument("--use_hierarchical_codebook", action="store_true")
    parser.add_argument("--no_hierarchical_codebook", action="store_true")
    parser.add_argument("--use_kinematic_decoder", action="store_true")
    parser.add_argument("--no_kinematic_decoder", action="store_true")
    parser.add_argument("--enable_hand_residual", action="store_true")
    parser.add_argument("--disable_hand_residual", action="store_true")

    parser.add_argument("--weighted_recon_loss_weight", type=float, default=None)
    parser.add_argument("--bone_length_loss_weight", type=float, default=None)
    parser.add_argument("--temporal_velocity_loss_weight", type=float, default=None)
    parser.add_argument("--temporal_accel_loss_weight", type=float, default=None)

    parser.add_argument("--vqvae_num_epochs", type=int, default=None)
    parser.add_argument("--vqvae_learning_rate", type=float, default=None)


def add_gpt_experiment_args(parser: argparse.ArgumentParser):
    add_common_data_args(parser)
    parser.add_argument("--gpt_num_epochs", type=int, default=None)
    parser.add_argument("--gpt_learning_rate", type=float, default=None)
    parser.add_argument("--gpt_layers", type=int, default=None)
    parser.add_argument("--gpt_hidden_dim", type=int, default=None)


def apply_vqvae_overrides(cfg, args: argparse.Namespace):
    _validate_bool_pair(args, "use_hierarchical_codebook", "no_hierarchical_codebook")
    _validate_bool_pair(args, "use_kinematic_decoder", "no_kinematic_decoder")
    _validate_bool_pair(args, "enable_hand_residual", "disable_hand_residual")

    cfg.apply_overrides(
        {
            "data_root": args.data_root,
            "dataset_name": args.dataset_name,
            "batch_size": args.batch_size,
            "val_batch_size": args.val_batch_size,
            "model_max_seq_len": args.model_max_seq_len,
            "embedding_dim": args.embedding_dim,
            "codebook_size": args.codebook_size,
            "downsample_rate": args.downsample_rate,
            "motion_encoder_layers": args.motion_encoder_layers,
            "motion_decoder_layers": args.motion_decoder_layers,
            "motion_encoder_hidden_dim": args.motion_encoder_hidden_dim,
            "motion_decoder_hidden_dim": args.motion_decoder_hidden_dim,
            "weighted_recon_loss_weight": args.weighted_recon_loss_weight,
            "bone_length_loss_weight": args.bone_length_loss_weight,
            "temporal_velocity_loss_weight": args.temporal_velocity_loss_weight,
            "temporal_accel_loss_weight": args.temporal_accel_loss_weight,
            "vqvae_num_epochs": args.vqvae_num_epochs,
            "vqvae_learning_rate": args.vqvae_learning_rate,
        }
    )

    if args.use_hierarchical_codebook:
        cfg.use_hierarchical_codebook = True
    if args.no_hierarchical_codebook:
        cfg.use_hierarchical_codebook = False

    if args.use_kinematic_decoder:
        cfg.use_kinematic_decoder = True
    if args.no_kinematic_decoder:
        cfg.use_kinematic_decoder = False

    if args.enable_hand_residual:
        cfg.enable_hand_residual = True
    if args.disable_hand_residual:
        cfg.enable_hand_residual = False

    cfg.refresh_dependent_values()


def apply_gpt_overrides(cfg, args: argparse.Namespace):
    cfg.apply_overrides(
        {
            "data_root": args.data_root,
            "dataset_name": args.dataset_name,
            "batch_size": args.batch_size,
            "val_batch_size": args.val_batch_size,
            "gpt_num_epochs": args.gpt_num_epochs,
            "gpt_learning_rate": args.gpt_learning_rate,
            "gpt_layers": args.gpt_layers,
            "gpt_hidden_dim": args.gpt_hidden_dim,
        }
    )
