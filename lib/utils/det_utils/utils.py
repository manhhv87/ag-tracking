from collections import defaultdict, deque
import datetime
import errno
import os
import pickle
import time

import torch
import torch.distributed as dist

from . import transforms as T


class SmoothedValue(object):
    """
    Track a series of numeric values and provide smoothed summaries over a
    recent window as well as over the entire series.

    Window stats (median, avg, max, value) are computed from a bounded deque
    of the last `window_size` updates, while `global_avg` is computed from a
    running sum and count (optionally synchronized across distributed ranks).
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        
        # Keep last `window_size` values for local (window) statistics
        self.deque = deque(maxlen=window_size)

        # Track global sum and count for global average
        self.total = 0.0
        self.count = 0

        # Default string format used by __str__
        self.fmt = fmt

    def update(self, value, n=1):
        """
        Add a new observation.

        Parameters
        ----------
        value : float
            The numeric value to record (e.g., loss for a batch).
        n : int, default 1
            How many samples this value represents (affects global average).
        """
        self.deque.append(value)     # add to window buffer
        self.count += n              # grow the global count
        self.total += value * n      # accumulate the global sum

    def synchronize_between_processes(self):
        """
        Synchronize `total` and `count` across distributed processes.

        Important
        ---------
        This does **not** synchronize the window `deque` (only running stats),
        so `median`/`avg` remain local while `global_avg` becomes consistent
        across ranks after synchronization.
        """
        if not is_dist_avail_and_initialized():
            return
        
        # Pack [count, total] into a CUDA tensor and all-reduce (sum across ranks)
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """Median of values in the current window."""
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """Mean of values in the current window."""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """Mean over all values ever seen (`total` / `count`)."""
        return self.total / self.count

    @property
    def max(self):
        """Maximum value in the current window."""
        return max(self.deque)

    @property
    def value(self):
        """Most recently added value."""
        return self.deque[-1]

    def __str__(self):
        """Human-readable summary string using `fmt`."""
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(data):
    """
    Gather arbitrary **picklable** Python data from all distributed processes.

    Why
    ---
    Native collectives require tensors of equal shapes. Here we:
      1) pickle Python objects to raw bytes,
      2) share each rank's byte size, pad to the max size,
      3) all_gather byte tensors,
      4) unpickle back to Python objects.

    Parameters
    ----------
    data : Any picklable object

    Returns
    -------
    list
        A list of length `world_size`, containing one object per rank
        (including the local rank’s object).
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # Serialize to bytes and wrap into a CUDA ByteTensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # First, gather sizes so we know how much to pad per rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # Prepare receive buffers (all same max length), pad local if needed
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    
    # Gather the padded tensors
    dist.all_gather(tensor_list, tensor)

    # Trim to original sizes and unpickle
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    All-reduce a dict of **scalar tensors** across distributed processes.

    Parameters
    ----------
    input_dict : dict[str, torch.Tensor]
        Keys are metric names; values must be scalar tensors.
    average : bool, default True
        If True, divide by `world_size` after summation (mean); else keep sums.

    Returns
    -------
    dict[str, torch.Tensor]
        A new dict with the same keys mapped to reduced tensors.

    Notes
    -----
    Keys are sorted to ensure consistent stacking order across ranks.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []        
        for k in sorted(input_dict.keys()): # fixed order across ranks
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0) # [N] tensor of scalars
        dist.all_reduce(values)             # sum across all ranks
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    """
    Keep and print multiple `SmoothedValue` meters, and provide a `log_every`
    wrapper to time data loading/iteration and estimate ETA during loops.

    Example
    -------
    >>> logger = MetricLogger(delimiter="  ")
    >>> for images, targets in logger.log_every(data_loader, print_freq=50, header="Train:"):
    ...     # training loop body...
    """

    def __init__(self, delimiter="\t"):
        # Auto-creates a new SmoothedValue when accessing a missing key
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """
        Update meters via keyword args, e.g. `update(loss=0.4, lr=1e-4)`.
        Tensors are converted to Python numbers.
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()    # convert scalar tensor to float
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """Allow attribute-style access to meters, e.g. `logger.loss`."""
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        """
        Return a single-line summary like:
        "loss: 0.1234 (0.4567)\tlr: 0.0010 (0.0010)"
        where each meter uses its own format string.
        """
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """Synchronize meters' running totals/counts across ranks."""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """Add a pre-created SmoothedValue under `name`."""
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        Wrap an iterable to:
          - time data loading (`data_time`) and iteration (`iter_time`)
          - estimate ETA from global average of iter_time
          - print a formatted log every `print_freq` steps (and at the end)

        Yields
        ------
        Items from the original iterable.
        """
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()

        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")

        # Pretty width for "[ i/len ]" display
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = self.delimiter.join(
            [
                header,
                "[{0" + space_fmt + "}/{1}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
                "max mem: {memory:.0f}",
            ]
        )
        MB = 1024.0 * 1024.0

        for obj in iterable:
            # Measure data loading time since the previous loop end
            data_time.update(time.time() - end)

            # Yield the batch/item to the caller (training/eval step)
            yield obj

            # Measure iteration (compute) time up to here
            iter_time.update(time.time() - end)

            # Print on schedule or at the very last iteration
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    log_msg.format(
                        i,
                        len(iterable),
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB,
                    )
                )
            i += 1
            end = time.time()

        # Final summary of the whole loop
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def collate_fn(batch):
    """
    Simple collate: transpose a list of tuples into a tuple of lists.

    Example
    -------
    Input:  [(img1, target1), (img2, target2)]
    Output: ((img1, img2), (target1, target2))
    """
    return tuple(zip(*batch))


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """
    Build a LambdaLR scheduler that linearly scales LR from `warmup_factor`
    to 1.0 over `warmup_iters` steps, then stays at 1.0.

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
    """
    def f(x):
        # After warmup, keep LR multiplier at 1.0
        if x >= warmup_iters:
            return 1
        
        # Linearly interpolate from warmup_factor -> 1.0
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    """
    Create a directory (and parents) if it doesn't already exist.

    Swallows the EEXIST error (directory already exists) and re-raises others.
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    Override built-in `print` so only the master process prints
    (unless forced with `force=True`) — keeps logs tidy under DDP.
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """
    Return True if torch.distributed is both available and initialized.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """
    Number of processes participating in the current distributed job (>= 1).
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """
    Rank (ID) of the current process in [0 .. world_size-1].
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """
    Convenience: True iff this process is rank 0 (master), else False.
    """
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    """
    Save a file only from the main process. Non-master ranks no-op to avoid
    redundant writes or clobbering.
    """
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    """
    Initialize torch.distributed according to environment variables or
    provided arguments, and set up printing for the master process only.

    Expected attributes on `args` (common patterns):
      - args.rank (int): global rank
      - args.world_size (int): total processes
      - args.dist_url (str): init method URL (e.g., "env://", "tcp://...")
      - args.dist_backend (str): backend, typically "nccl" for GPU
      - args.gpu (int): local GPU index on the node (optional)

    This function attempts to infer rank/world_size/gpu from environment
    variables commonly used by `torchrun` or SLURM if not already set.
    """
    if getattr(args, "dist_url", None) is None:
        # Default to environment-based init when possible
        args.dist_url = "env://"
    
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", -1)))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if rank != -1 and world_size != -1:
        # Launched with torchrun/torch.distributed.run
        args.rank = rank
        args.world_size = world_size
        args.gpu = local_rank if local_rank != -1 else 0
    else:
        # Fallback to single-process defaults if not provided
        args.rank = getattr(args, "rank", 0)
        args.world_size = getattr(args, "world_size", 1)
        # Try to read a user-provided GPU index; fall back to 0
        args.gpu = getattr(args, "gpu", 0)

    # Initialize process group if running distributed
    if args.world_size > 1:
        torch.cuda.set_device(args.gpu)
        dist_backend = getattr(args, "dist_backend", "nccl")
        dist.init_process_group(
            backend=dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        torch.distributed.barrier()

    # Ensure only master process prints by default
    setup_for_distributed(args.rank == 0)


def get_transform(train):
    """
    Build a basic image transform pipeline.

    Behavior
    --------
    - Always converts PIL images to torch tensors (`ToTensor`).
    - If `train` is True, also applies random horizontal flip with p=0.5
      to both the image and its targets (data augmentation).
    """
    transforms = []

    # Convert PIL image to a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # During training, randomly flip images (and targets) as augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
