import argparse
import logging
import sys
from pathlib import Path
from typing import List

myloglevel = "ERROR"

# Configure logging with timestamps
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
    help="Number of parallel processes (default: 1)",
)
parser.add_argument(
    "--tune", action="store_true", help="Do tuning [recommended] (default: False)"
)
parser.add_argument(
    "--tune_down_sample",
    required=False,
    default=0,
    type=float,
    help="Downsample stack during tuning: increases this value from 0 up to 1 if tuning is slow.",
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

from basicpy import BaSiC


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
):
    logger = logging.getLogger(__name__)
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

    logger.info(f"{batch_prefix} Starting batch with {len(file_list)} files")

    stack_list = []
    meta_list = []
    for idx, f in enumerate(file_list, 1):
        logger.info(f"{batch_prefix} Reading file {idx}/{len(file_list)}: {f.name}")
        img, meta = read_img_stack(f, quiet=quiet)
        stack_list.append(img)
        meta_list.append(meta)

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
            max_iterations=1000,
            max_reweight_iterations=50,
            max_reweight_iterations_baseline=25,
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
                    stack_full[0, z_start:z_end, i, :, :].squeeze(),
                    early_stop=True,
                    n_iter=10,
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
                    max_iterations=1000,
                    max_reweight_iterations=50,
                    max_reweight_iterations_baseline=25,
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
            ].squeeze()
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
                .transform(stack[0, :, j, :, :].squeeze())
                .astype(np.uint16)
            )
        print()
        output_fname = input_fname.parent / f"corrected_{input_fname.stem}.tif"
        logger.info(f"{batch_prefix} Saving corrected stack: {output_fname.name}")
        save_img_stack(str(output_fname), stack_correct, meta)

    logger.info(f"{batch_prefix} Batch processing completed")
    javabridge.kill_vm()


def main():
    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    logger.info("Starting flat-field correction")

    img_files = find_imgs(args.path, args.ext)
    logger.info(
        f"Found {len(img_files)} files with extension '{args.ext}' in {args.path}"
    )
    n_imgs_per_batch = min(len(img_files), args.num_per_batch)
    # Create batches of files
    batches = [
        img_files[k : k + n_imgs_per_batch]
        for k in range(0, len(img_files), n_imgs_per_batch)
    ]
    logger.info(
        f"Created {len(batches)} batches with up to {n_imgs_per_batch} files per batch"
    )
    if args.num_process > 1:
        from functools import partial
        from multiprocessing import Pool

        logger.info(
            f"Running in multiprocessing mode with {args.num_process} processes"
        )
        # In multiprocessing mode, disable tqdm to avoid conflicts
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
        logger.info("Running in single-process mode")
        for batch_id, batch in enumerate(batches, 1):
            basic_worker(
                batch,
                batch_id=batch_id,
                tune=args.tune,
                tune_down_sample=args.tune_down_sample,
                quiet=False,
                parallel=False,
            )
    logger.info("Flat-field correction completed successfully")


if __name__ == "__main__":
    main()
