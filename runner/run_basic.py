import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

myloglevel = "ERROR"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue


def set_envs(num_process: int):
    logger = logging.getLogger(__name__)
    total_cpus = os.cpu_count() or 1
    threads_per_process = max(1, total_cpus // num_process)

    os.environ["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen=true "
        f"intra_op_parallelism_threads={threads_per_process}"
    )
    os.environ["OMP_NUM_THREADS"] = str(threads_per_process)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_process)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_process)

    logger.info(
        f"CPU allocation: {total_cpus} total CPUs, {num_process} processes, "
        f"{threads_per_process} threads per process"
    )


parser = argparse.ArgumentParser(description="Run flat-field correction using BaSiC")
parser.add_argument(
    "--path", required=True, help="Root folder path where images are stored"
)
parser.add_argument("--ext", required=True, help="File extension (e.g. vsi)")
parser.add_argument(
    "--num_per_batch",
    required=False,
    default=24,
    type=positive_int,
    help="Number of stacks to process per batch",
)
parser.add_argument(
    "--num_process",
    required=False,
    default=1,
    type=positive_int,
    help="Number of parallel processes for CPU fallback mode (default: 1). Auto-detects GPU and uses single-process GPU mode if available.",
)
parser.add_argument(
    "--tune",
    action="store_true",
    help="Do tuning [disabled, haven't tested yet on basic v2] (default: False)",
)
parser.add_argument(
    "--tune_down_sample",
    required=False,
    default=0,
    type=float,
    help="Downsample stack during tuning [disabled, basic v2 tuning not tested yet]: increases this value from 0 up to 1 if tuning is slow.",
)
parser.add_argument(
    "--wall_time",
    required=False,
    default=0,
    type=float,
    help="Wall time limit per batch in minutes. If exceeded, retry with halved iteration params. 0 = no limit (default: 0)",
)
parser.add_argument(
    "--max_retries",
    required=False,
    default=3,
    type=positive_int,
    help="Max retries with reduced params when wall time is exceeded (default: 3)",
)
# print help if nothing provided
if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

# move those here to speed up doc loop up
import bioformats
import javabridge
import numpy as np
import tifffile
from tqdm import trange


def check_gpu_available() -> tuple[bool, str]:
    """Check if GPU (CUDA or Apple MPS) is available for PyTorch.

    Returns:
        tuple: (is_available: bool, device_type: str)
               device_type is 'cuda', 'mps', or 'cpu'
    """
    try:
        import torch

        if torch.cuda.is_available():
            return True, "cuda"
        elif torch.backends.mps.is_available():
            return True, "mps"
        else:
            return False, "cpu"
    except ImportError:
        return False, "cpu"


def get_processing_mode(args_num_process: int) -> tuple[str, bool, str]:
    """Determine processing mode based on GPU availability and user args.

    Args:
        args_num_process: Number of processes requested by user

    Returns:
        tuple: (mode: str, has_gpu: bool, device: str)
               mode is 'gpu', 'multiprocessing', or 'single'
    """
    has_gpu, device = check_gpu_available()

    if has_gpu:
        return "gpu", True, device
    elif args_num_process > 1:
        return "multiprocessing", False, "cpu"
    else:
        return "single", False, "cpu"


def read_img_stack(path, quiet=False):
    meta = bioformats.get_omexml_metadata(str(path))
    meta = bioformats.omexml.OMEXML(meta).image(0)
    size_t = meta.Pixels.get_SizeT()
    size_z = meta.Pixels.get_SizeZ()
    size_y = meta.Pixels.get_SizeY()
    size_x = meta.Pixels.get_SizeX()
    size_c = meta.Pixels.get_SizeC()

    # not sure how to get time interval, leave it for now
    pixel_size_z = meta.Pixels.get_PhysicalSizeZ()
    pixel_size_y = meta.Pixels.get_PhysicalSizeY()
    pixel_size_x = meta.Pixels.get_PhysicalSizeX()

    unit_xy = meta.Pixels.get_PhysicalSizeXUnit()
    unit_z = meta.Pixels.get_PhysicalSizeZUnit()

    stack = np.zeros((size_t, size_z, size_c, size_y, size_x), dtype=np.uint16)
    with bioformats.ImageReader(str(path)) as reader:
        if not quiet:
            for t in trange(size_t, desc=f"Read {path.name} T-frames", position=0):
                for z in trange(
                    size_z, desc=f"Read {path.name} Z-slices (t = {t + 1})", leave=False
                ):
                    for c in range(size_c):
                        img = reader.read(t=t, z=z, c=c)
                        if img.dtype == np.float32 or img.dtype == np.float64:
                            img = (img * 65535).astype(np.uint16)
                        stack[t, z, c, ...] = img
        else:
            # Quiet mode: no progress bars, just read
            for t in range(size_t):
                for z in range(size_z):
                    for c in range(size_c):
                        img = reader.read(t=t, z=z, c=c)
                        if img.dtype == np.float32 or img.dtype == np.float64:
                            img = (img * 65535).astype(np.uint16)
                        stack[t, z, c, ...] = img

    meta = {
        "pixel_size_x": pixel_size_x,
        "pixel_size_y": pixel_size_y,
        "pixel_size_z": pixel_size_z,
        "unit_xy": unit_xy,
        "unit_z": unit_z,
    }
    return stack, meta


def save_img_stack(path: str, img, meta):
    px = meta.get("pixel_size_x")
    py = meta.get("pixel_size_y")

    if px is not None and py is not None:
        resolution = (1.0 / px, 1.0 / py)
    else:
        resolution = None

    tifffile.imwrite(
        path,
        img,
        imagej=True,
        resolution=resolution,
        metadata={
            "axes": "TZCYX",
            "spacing": meta["pixel_size_z"],
        },
    )


def find_imgs(root_path, extension):
    path = Path(root_path)
    if not path.is_dir():
        raise ValueError(f"{root_path} folder does not exist")

    files = [f for f in path.glob(f"*{extension}")]
    return sorted(files)


def basic_worker(
    file_list: List,
    batch_id: int,
    tune: bool,
    tune_down_sample: float,
    quiet: bool = False,
    parallel: bool = False,
    iter_params: dict = None,
    reading_done_event=None,
    manage_jvm: bool = True,
):
    # Delay the import so spawned workers can import this module safely.
    from basicpy import BaSiC

    logger = logging.getLogger(__name__)
    if manage_jvm:
        javabridge.start_vm(class_path=bioformats.JARS)
    rootLoggerName = javabridge.get_static_field(
        "org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;"
    )
    rootLogger = javabridge.static_call(
        "org/slf4j/LoggerFactory",
        "getLogger",
        "(Ljava/lang/String;)Lorg/slf4j/Logger;",
        rootLoggerName,
    )
    logLevel = javabridge.get_static_field(
        "ch/qos/logback/classic/Level", myloglevel, "Lch/qos/logback/classic/Level;"
    )
    javabridge.call(
        rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel
    )
    batch_prefix = f"[Batch {batch_id}]"

    if iter_params is None:
        iter_params = {
            "max_iterations": 1000,
            "max_reweight_iterations": 50,
            "max_reweight_iterations_baseline": 25,
            "optimization_tol": 1e-3,
            "reweighting_tol": 1e-2,
        }

    logger.info(
        f"{batch_prefix} Starting batch with {len(file_list)} files "
        f"(max_iter={iter_params['max_iterations']}, "
        f"max_reweight={iter_params['max_reweight_iterations']}, "
        f"opt_tol={iter_params['optimization_tol']:.1e}, "
        f"reweight_tol={iter_params['reweighting_tol']:.1e})"
    )

    try:
        stack_list = []
        meta_list = []
        for idx, f in enumerate(file_list, 1):
            logger.info(f"{batch_prefix} Reading file {idx}/{len(file_list)}: {f.name}")
            img, meta = read_img_stack(f, quiet=quiet)
            stack_list.append(img)
            meta_list.append(meta)

        if reading_done_event is not None:
            reading_done_event.set()

        # concatenate along z
        stack_full = np.concatenate(stack_list, 1)
        logger.info(
            f"{batch_prefix} Concatenated stack shape: {stack_full.shape} "
            f"(T={stack_full.shape[0]}, Z={stack_full.shape[1]}, C={stack_full.shape[2]})"
        )

        basic_models = []
        # fit for each channel
        for i in range(stack_full.shape[2]):
            logger.info(f"{batch_prefix} Fitting channel {i + 1}/{stack_full.shape[2]}")
            basic = BaSiC(
                get_darkfield=True,
                **iter_params,
            )
            if tune:
                try:
                    # take center portion of Z-stack for tuning (faster for large stacks)
                    z_start = int(stack_full.shape[1] * tune_down_sample / 2)
                    z_end = int(stack_full.shape[1] * (1 - tune_down_sample / 2))
                    z_slices_used = z_end - z_start
                    logger.info(
                        f"{batch_prefix} Tuning channel {i + 1} using Z-slices {z_start}:{z_end} "
                        f"({z_slices_used}/{stack_full.shape[1]} slices)"
                    )
                    basic.autotune(
                        stack_full[0, z_start:z_end, i, :, :],
                        init_params={
                            "smoothness_flatfield": 2,
                            "smoothness_darkfield": 4,
                            "sparse_cost_darkfield": 0.01,
                        },
                    )
                    logger.info(f"{batch_prefix} Tuning completed for channel {i + 1}")
                except RuntimeError as e:
                    # if tuning failed, use my parameters
                    logger.warning(
                        f"{batch_prefix} Tuning failed for channel {i + 1}: {e}. "
                        "Using default parameters."
                    )
                    basic = BaSiC(
                        get_darkfield=True,
                        smoothness_darkfield=4,
                        smoothness_flatfield=2,
                        sparse_cost_darkfield=0.01,
                        **iter_params,
                    )
            # TODO: supports time series
            logger.info(f"{batch_prefix} Fitting BaSiC model for channel {i + 1}")
            basic.fit(
                stack_full[
                    0,
                    :,
                    i,
                    :,
                    :,
                ]
            )
            logger.info(f"{batch_prefix} Fitting completed for channel {i + 1}")
            basic_models.append(basic)

        for i in range(len(stack_list)):
            stack = stack_list[i]
            meta = meta_list[i]
            stack_correct = np.zeros_like(stack, dtype=stack.dtype)
            input_fname = file_list[i]
            logger.info(
                f"{batch_prefix} Correcting stack {i + 1}/{len(stack_list)}: {input_fname.name}"
            )
            for j in range(stack.shape[2]):
                print(".", end="", flush=True)
                stack_correct[0, :, j, :, :] = (
                    basic_models[j]
                    .transform(stack[0, :, j, :, :], use_tqdm=False)
                    .astype(np.uint16)
                )
            print()
            output_fname = input_fname.parent / f"corrected_{input_fname.stem}.tif"
            logger.info(f"{batch_prefix} Saving corrected stack: {output_fname.name}")
            save_img_stack(str(output_fname), stack_correct, meta)

        logger.info(f"{batch_prefix} Batch processing completed")
    finally:
        if manage_jvm:
            javabridge.kill_vm()


def run_batch_with_timeout(
    batch,
    batch_id,
    tune,
    tune_down_sample,
    quiet,
    parallel,
    wall_time_minutes,
    max_retries,
):
    from multiprocessing import Event, Process

    logger = logging.getLogger(__name__)

    iter_params = {
        "max_iterations": 1000,
        "max_reweight_iterations": 50,
        "max_reweight_iterations_baseline": 25,
        "optimization_tol": 1e-3,
        "reweighting_tol": 1e-2,
    }

    for attempt in range(max_retries + 1):
        if attempt > 0:
            iter_params = {
                "max_iterations": max(50, iter_params["max_iterations"] // 2),
                "max_reweight_iterations": max(
                    2, iter_params["max_reweight_iterations"] // 2
                ),
                "max_reweight_iterations_baseline": max(
                    2, iter_params["max_reweight_iterations_baseline"] // 2
                ),
                "optimization_tol": iter_params["optimization_tol"] * 2,
                "reweighting_tol": iter_params["reweighting_tol"] * 2,
            }
            logger.warning(
                f"[Batch {batch_id}] Retry {attempt}/{max_retries} with reduced params: "
                f"max_iter={iter_params['max_iterations']}, "
                f"max_reweight={iter_params['max_reweight_iterations']}, "
                f"opt_tol={iter_params['optimization_tol']:.1e}, "
                f"reweight_tol={iter_params['reweighting_tol']:.1e}"
            )

        reading_done = Event()
        p = Process(
            target=basic_worker,
            args=(batch, batch_id, tune, tune_down_sample, quiet, parallel),
            kwargs={"iter_params": iter_params, "reading_done_event": reading_done},
        )
        p.start()

        while not reading_done.wait(timeout=10):
            if not p.is_alive():
                break

        if p.is_alive():
            if wall_time_minutes > 0:
                p.join(timeout=wall_time_minutes * 60)
            else:
                p.join()

        if not p.is_alive():
            if p.exitcode == 0:
                if attempt > 0:
                    logger.info(
                        f"[Batch {batch_id}] Completed on retry attempt {attempt}"
                    )
                return
            else:
                logger.error(
                    f"[Batch {batch_id}] Process crashed with exit code {p.exitcode}"
                )
                return

        logger.warning(
            f"[Batch {batch_id}] Exceeded {wall_time_minutes} min wall time "
            f"(attempt {attempt + 1}/{max_retries + 1})"
        )
        p.terminate()
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
            p.join()

    logger.error(
        f"[Batch {batch_id}] Failed after {max_retries + 1} attempts, skipping."
    )


def main():
    logger = logging.getLogger(__name__)
    args = parser.parse_args()
    processing_mode, _, device = get_processing_mode(args.num_process)

    if processing_mode == "multiprocessing":
        set_envs(args.num_process)
        logger.info(f"CPU mode: Using {args.num_process} process(es)")
    elif processing_mode == "single":
        logger.info("CPU mode: Using single process")
    else:
        logger.info(f"GPU mode: Using {device} device with PyTorch backend")

    logger.info("Starting flat-field correction")

    img_files = find_imgs(args.path, args.ext)
    logger.info(
        f"Found {len(img_files)} files with extension '{args.ext}' in {args.path}"
    )
    n_imgs_per_batch = min(len(img_files), args.num_per_batch)
    batches = [
        img_files[k : k + n_imgs_per_batch]
        for k in range(0, len(img_files), n_imgs_per_batch)
    ]
    logger.info(
        f"Created {len(batches)} batches with up to {n_imgs_per_batch} files per batch"
    )

    if processing_mode == "gpu":
        logger.info(f"GPU mode: Processing {len(batches)} batches on {device}")
        javabridge.start_vm(class_path=bioformats.JARS)
        try:
            for batch_id, batch in enumerate(batches, 1):
                basic_worker(
                    batch,
                    batch_id=batch_id,
                    tune=args.tune,
                    tune_down_sample=args.tune_down_sample,
                    quiet=False,
                    parallel=False,
                    manage_jvm=False,
                )
        finally:
            javabridge.kill_vm()
    elif processing_mode == "multiprocessing":
        if args.wall_time > 0:
            logger.info(
                f"Wall time enabled: {args.wall_time} min per batch, "
                f"max {args.max_retries} retries with halved parameters"
            )
            from concurrent.futures import ThreadPoolExecutor

            logger.info(
                f"Running in parallel timeout mode with {args.num_process} processes"
            )
            with ThreadPoolExecutor(max_workers=args.num_process) as executor:
                futures = [
                    executor.submit(
                        run_batch_with_timeout,
                        batch,
                        batch_id,
                        tune=args.tune,
                        tune_down_sample=args.tune_down_sample,
                        quiet=True,
                        parallel=True,
                        wall_time_minutes=args.wall_time,
                        max_retries=args.max_retries,
                    )
                    for batch_id, batch in enumerate(batches, 1)
                ]
                for f in futures:
                    f.result()
        else:
            from functools import partial
            from multiprocessing import Pool

            logger.info(
                f"Running in multiprocessing mode with {args.num_process} processes"
            )
            worker = partial(
                basic_worker,
                tune=args.tune,
                tune_down_sample=args.tune_down_sample,
                quiet=True,
                parallel=True,
            )
            with Pool(processes=args.num_process) as pool:
                pool.starmap(
                    worker,
                    [(b, i) for i, b in enumerate(batches)],
                )
    else:
        logger.info("Running in single-process CPU mode")
        if args.wall_time > 0:
            for batch_id, batch in enumerate(batches, 1):
                run_batch_with_timeout(
                    batch,
                    batch_id,
                    tune=args.tune,
                    tune_down_sample=args.tune_down_sample,
                    quiet=False,
                    parallel=False,
                    wall_time_minutes=args.wall_time,
                    max_retries=args.max_retries,
                )
        else:
            javabridge.start_vm(class_path=bioformats.JARS)
            try:
                for batch_id, batch in enumerate(batches, 1):
                    basic_worker(
                        batch,
                        batch_id=batch_id,
                        tune=args.tune,
                        tune_down_sample=args.tune_down_sample,
                        quiet=False,
                        parallel=False,
                        manage_jvm=False,
                    )
            finally:
                javabridge.kill_vm()
    logger.info("Flat-field correction completed successfully")


if __name__ == "__main__":
    main()
