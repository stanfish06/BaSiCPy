import argparse
from pathlib import Path

import bioformats
import javabridge
import numpy as np
import tifffile
from tqdm import trange

from basicpy import BaSiC

myloglevel = "ERROR"


def read_img_stack(path):
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
        for t in trange(size_t, desc=f"Read {path.name} T-frames", position=0):
            for z in trange(
                size_z, desc=f"Read {path.name} Z-slices (t = {t + 1})", leave=False
            ):
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
    resolution_x = 1.0 / meta["pixel_size_x"] if meta["pixel_size_x"] else None
    resolution_y = 1.0 / meta["pixel_size_y"] if meta["pixel_size_y"] else None
    tifffile.imwrite(
        path,
        img,
        imagej=True,
        resolution=(resolution_x, resolution_y),
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


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue


def main():
    parser = argparse.ArgumentParser(
        description="Run flat-field correction using BaSiC"
    )
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
        "--tune", action="store_true", help="Do tuning [recommended] (default: False)"
    )
    parser.add_argument(
        "--tune_down_sample",
        required=False,
        default=0,
        type=float,
        help="Downsample stack during tuning: increaes this value from 0 up to 1 if tuning is slow.",
    )
    args = parser.parse_args()

    print("start flat-field correction")
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
    img_files = find_imgs(args.path, args.ext)
    n_imgs_per_batch = min(len(img_files), args.num_per_batch)
    for k in range(0, len(img_files), n_imgs_per_batch):
        img_files_sub = img_files[k : (k + n_imgs_per_batch)]
        stack_list = []
        meta_list = []
        for f in img_files_sub:
            img, meta = read_img_stack(f)
            stack_list.append(img)
            meta_list.append(meta)
        # concatenate along z
        stack_full = np.concatenate(stack_list, 1)
        basic_models = []
        # fit for each channel
        for i in range(stack_full.shape[2]):
            print(f"Fit/Tune: channel {i + 1}")
            basic = BaSiC(
                get_darkfield=True,
                max_iterations=1000,
                max_reweight_iterations=50,
                max_reweight_iterations_baseline=25,
            )
            if args.tune:
                try:
                    basic.autotune(
                        stack_full[
                            0,
                            int(stack_full.shape[1] * args.tune_down_sample / 2) : int(
                                stack_full.shape[1] * (1 - args.tune_down_sample / 2)
                            ),
                            i,
                            :,
                            :,
                        ].squeeze(),
                        early_stop=True,
                        n_iter=10,
                        init_params={
                            "smoothness_flatfield": 2,
                            "smoothness_darkfield": 4,
                            "sparse_cost_darkfield": 0.01,
                        },
                    )
                except RuntimeError:
                    # if tuning failed, use my parameters
                    basic = BaSiC(
                        get_darkfield=True,
                        smoothness_darkfield=4,
                        smoothness_flatfield=2,
                        sparse_cost_darkfield=0.01,
                        max_iterations=1000,
                        max_reweight_iterations=50,
                        max_reweight_iterations_baseline=25,
                    )
                    print("Tuning failed. Proceed to fitting.")
            # TODO: supports time series
            basic.fit(
                stack_full[
                    0,
                    :,
                    i,
                    :,
                    :,
                ].squeeze()
            )
            basic_models.append(basic)
        for i in range(len(stack_list)):
            stack = stack_list[i]
            meta = meta_list[i]
            stack_correct = np.zeros_like(stack, dtype=stack.dtype)
            print(f"Correct: stack {i + 1}")
            for j in range(stack.shape[2]):
                print(".", end="", flush=True)
                stack_correct[0, :, j, :, :] = (
                    basic_models[j]
                    .transform(stack[0, :, j, :, :].squeeze())
                    .astype(np.uint16)
                )
            print()
            input_fname = img_files_sub[i]
            output_fname = input_fname.parent / f"corrected_{input_fname.stem}.tif"
            print(f"Save: {output_fname.name}")
            save_img_stack(str(output_fname), stack_correct, meta)
    javabridge.kill_vm()
    print("flat-field correction done")


if __name__ == "__main__":
    main()
