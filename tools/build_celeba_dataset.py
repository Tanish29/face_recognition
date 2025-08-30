import gdown as gd
import click
from pathlib import Path
import logging
from py7zr import SevenZipFile
from py7zr.callbacks import ExtractCallback
import os
from tqdm import tqdm
import multivolumefile

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ExtractProgress(ExtractCallback):
    def __init__(self, total_files, lim):
        self.pbar = tqdm(total=total_files, desc="Extracting")
        self.lim = lim

    def report_update(self, *args, **kwargs):
        self.pbar.update(1)
        if self.pbar.n == self.lim:
            log.info(f"Exiting, extracted {self.lim} images")
            os._exit(0)

    def report_start(self, *args, **kwargs):
        return super().report_start(*args, **kwargs)

    def report_end(self, *args, **kwargs):
        return super().report_end(*args, **kwargs)

    def report_postprocess(self, *args, **kwargs):
        return super().report_postprocess(*args, **kwargs)

    def report_start_preparation(self, *args, **kwargs):
        return super().report_start_preparation(*args, **kwargs)

    def report_warning(self, *args, **kwargs):
        return super().report_warning(*args, **kwargs)


@click.command()
@click.option(
    "--out-dir",
    type=str,
    default="dataset/celeba",
    help="Output directory to save the dataset.",
)
@click.option(
    "-n",
    type=int,
    default=None,
    help="Number of images to download. Default is None (download all).",
)
def build_celeba_dataset(out_dir, n):
    """
    Download the CelebA dataset from Google Drive.

    Dataset:
    liu2015faceattributes,
    author = {Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang},
    """
    out_dir = Path(out_dir)
    # download necessary contents from drive
    zip_dir = out_dir / "7zs"
    zip_dir.mkdir(parents=True, exist_ok=True)
    img_folder = "0B7EVK8r0v71peklHb0pGdDl6R28?resourcekey=0-f5cwz-nTIQC3KsBn3wFn7A"
    gd.download_folder(
        id=img_folder,
        output=zip_dir.as_posix(),
        resume=True,
    )
    anno_dir = out_dir / "annotations"
    anno_dir.mkdir(parents=True, exist_ok=True)
    anno_folder = "0B7EVK8r0v71pOC0wOVZlQnFfaGs?resourcekey=0-pEjrQoTrlbjZJO2UL8K_WQ"
    gd.download_folder(
        id=anno_folder,
        output=anno_dir.as_posix(),
        resume=True,
    )
    # unzip the multi volume files
    if not (out_dir / "img_celeba").exists():
        log.info("Extracting Images...")
        example_7z = zip_dir / "img_celeba.7z"
        with multivolumefile.open(example_7z, mode="rb") as target_archive:
            with SevenZipFile(target_archive, "r") as archive:
                num_imgs = len(archive.getnames())
                archive.extractall(path=out_dir, callback=ExtractProgress(num_imgs, n))

    log.info(f"CelebA dataset built and saved to {out_dir}")


if __name__ == "__main__":
    build_celeba_dataset()
