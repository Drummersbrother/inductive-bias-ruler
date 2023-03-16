from src import models, trainer

if __name__ == "__main__":
    model_config = dict(
        dataset_name="MNIST",
        number_of_classes=10,
        conv_layers=[
            dict(out_channels=8, kernel_size=5, stride=1)
        ],
        linear_layers=[
            dict(out_features=256),
            dict(out_features=64),
            dict(out_features="num_classes")
        ],
        activation_fct=[
            dict(name="ReLU", kwargs={})
        ],
        post_activation_fct=[
            dict(name="MaxPool2d", kwargs=dict(kernel_size=2))
        ],
        dl_kwargs=dict(batch_size=4096),
        loss_function="CrossEntropyLoss",
        epochs=10,
        learning_rate=3e-4,
        optimizer=("Adam", dict()),
        learning_rate_scheduler=("CosineAnnealingWarmRestarts", dict(eta_min=1e-5, T_0=10, T_mult=1))
    )

    train_config = dict(
        val_check_interval=1.
    )

    model = models.BaseModule(model_config)

    trainer.train_model(model, trainer_kwargs=train_config)
