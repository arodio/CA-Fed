class Client:
    r"""Represents a client participating in the learning process

    Attributes
    ----------
    client_id:

    client_id: int

    trainer: Trainer

    device: str or torch.device

    train_loader: torch.utils.data.DataLoader

    val_loader: torch.utils.data.DataLoader

    test_loader: torch.utils.data.DataLoader

    train_iterator:

    local_steps: int

    metadata: dict

    logger: torch.utils.tensorboard.SummaryWriter

    """
    def __init__(
            self,
            client_id,
            local_steps,
            logger,
            trainer=None,
            train_loader=None,
            val_loader=None,
            test_loader=None,
    ):

        self.client_id = client_id

        self.trainer = trainer

        self.device = self.trainer.device

        if train_loader is not None:
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader

            self.num_samples = len(self.train_loader.dataset)

            self.train_iterator = iter(self.train_loader)

            self.is_ready = True

        else:
            self.is_ready = False

        self.local_steps = local_steps

        self.logger = logger

        self.metadata = dict()

        self.counter = 0

    def get_next_batch(self):
        """yields next batch from data generator

        Returns
        -------
            * Tuple(torch.tensor, torch.tensor): batch given as a tuple of tensors
        """

        try:
            batch = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_loader)
            batch = next(self.train_iterator)

        return batch

    def step(self, by_batch=False):
        """perform one local step

        Parameters
        ----------
        by_batch: bool

        Returns
        -------
            float: loss
        """
        self.counter += 1

        if by_batch:
            batch = self.get_next_batch()
            self.trainer.fit_batch(batch=batch)
        else:
            self.trainer.fit_epochs(
                loader=self.train_loader,
                n_epochs=self.local_steps
            )

    def write_logs(self, counter=None):
        if counter is None:
            counter = self.counter

        train_loss, train_metric = self.trainer.evaluate_loader(self.train_loader)
        test_loss, test_metric = self.trainer.evaluate_loader(self.test_loader)

        self.logger.add_scalar("Train/Loss", train_loss, counter)
        self.logger.add_scalar("Train/Metric", train_metric, counter)
        self.logger.add_scalar("Test/Loss", test_loss, counter)
        self.logger.add_scalar("Test/Metric", test_metric, counter)
        self.logger.flush()

        return train_loss, train_metric, test_loss, test_metric
