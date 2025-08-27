import bioformats
import numpy as np
import tifffile
import os
import javabridge
import argparse
from pathlib import Path
from basicpy import BaSiC
from tqdm import tqdm, trange
import sys

myloglevel = "ERROR"


def read_img_stack(path):
    print(f"Read: {path.name}")
    meta = bioformats.get_omexml_metadata(str(path))
    meta = bioformats.omexml.OMEXML(meta).image(0)
    size_z = meta.Pixels.SizeZ
    size_y = meta.Pixels.SizeY
    size_x = meta.Pixels.SizeX
    size_c = meta.Pixels.SizeC
    print(size_z)

    stack = np.zeros((size_z, size_y, size_x, size_c), dtype=np.uint16)
    with bioformats.ImageReader(str(path)) as reader:
        for z in trange(size_z):
            img = reader.read(z=z)
            if img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 65535).astype(np.uint16)
            stack[z] = img
    return stack


def find_imgs(root_path, extension):
    path = Path(root_path)
    if not path.is_dir():
        raise ValueError(f"{root_path} folder does not exist")

    files = [f for f in path.rglob(f"*{extension}")]
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
        default=4,
        help="Number of stacks to process per batch",
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
        for f in img_files_sub:
            stack_list.append(read_img_stack(f))
        stack_full = np.concatenate(stack_list, 0)
        basic_models = []
        for i in range(stack_full.shape[-1]):
            print(f"Fit: channel {i + 1}")
            basic = BaSiC(get_darkfield=True)
            basic.fit(stack_full[:, :, :, i])
            basic_models.append(basic)
        for i in range(len(stack_list)):
            stack = stack_list[i]
            stack_correct = np.zeros_like(stack, dtype=stack.dtype)
            print(f"Correct: stack {i + 1}")
            for j in range(stack.shape[-1]):
                print(".", end="")
                stack_correct[:, :, :, j] = (
                    basic_models[j].transform(stack[:, :, :, j]).astype(np.uint16)
                )
            print()
            input_fname = img_files[i]
            output_fname = input_fname.parent / f"{input_fname.stem}_corrected.tif"
            output_fname_original = input_fname.parent / f"{input_fname.stem}.tif"
            print(f"Save: {output_fname.name}")
            tifffile.imwrite(
                str(output_fname),
                stack_correct,
                metadata={"axes": "ZYXC", "mode": "composite"},
            )
            tifffile.imwrite(
                str(output_fname_original),
                stack,
                metadata={"axes": "ZYXC", "mode": "composite"},
            )
    javabridge.kill_vm()
    print("flat-field correction done")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.exit(1)
