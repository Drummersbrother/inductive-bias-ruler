from src import models, trainer, utils, PL_LOGS
from pprint import pprint
from tqdm import tqdm


def do_train():
    models_config = dict(
        n_ensembled=[50],
        dataset_name=["MNIST"],
        conv_blocks=[
            ((32, 5, 1, "valid"), (128, 3, 1, "valid"), (256, 3, 1, "valid")),
            ((8, 3, 1, "valid"), (16, 5, 1, "valid")),
            ((4, 3, 1, "valid"), (16, 3, 1, "valid"), (32, 3, 1, "valid")),
            tuple()
        ],
        linear_blocks=[
            (1024, 1024, "num_classes"),
            (1024, 512, 256, "num_classes"),
            (256, 64, "num_classes"),
            (64, 32, "num_classes"),
        ],
        activation_fcts=[
            (("ReLU", {}),)
        ],
        post_activation_fcts=[
            (("MaxPool2d", dict(kernel_size=2)),)
        ],
        dl_kwargs=[dict(batch_size=1024, num_workers=4)],
        loss_function=["CrossEntropyLoss"],
        epochs=[10],
        learning_rate=[3e-4],
        optimizer=[("Adam", dict())],
        learning_rate_scheduler=[("CosineAnnealingWarmRestarts", dict(eta_min=1e-5, T_0=10, T_mult=1))]
    )
    model_configs = utils.model_generator(
        **models_config
    )

    model_configs = [mcf for mcf in model_configs if sum((bool(mcf["conv_layers"]), bool(mcf["linear_layers"]))) >= 1]

    train_config = dict(
        check_val_every_n_epoch=1,
        log_every_n_steps=20,
        enable_model_summary=False,
        default_root_dir=PL_LOGS
    )
    to_train_models = [models.BaseModule(mcfg) for mcfg in model_configs]
    trained_models = []

    for model in tqdm(to_train_models, desc="Training multiple models"):
        trained_models.append(trainer.train_model(model, trainer_kwargs=train_config).cpu())
        del model

    print("Done")
    return trained_models
