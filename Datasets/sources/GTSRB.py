import csv
import pathlib
from typing import Any, Callable, Optional, Union

import PIL
from torchvision.datasets import VisionDataset

from torchvision.datasets.folder import make_dataset
from torchvision.datasets.utils import verify_str_arg


class GTSRB(VisionDataset):
    def __init__(
        self,
        root: Union[str, pathlib.Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root)
        self._target_folder = (
            self._base_folder / ("Training" if self._split == "train" else "Final_Test/Images")
        )


        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        if self._split == "train":
            samples = make_dataset(str(self._target_folder), extensions=(".ppm",))
        else:
            with open(self._base_folder / "GT-final_test.csv") as csv_file:
                samples = [
                    (str(self._target_folder / row["Filename"]), int(row["ClassId"]))
                    for row in csv.DictReader(csv_file, delimiter=";", skipinitialspace=True)
                ]

        self._samples = samples
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:

        path, target = self._samples[index]
        sample = PIL.Image.open(path).convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def _check_exists(self) -> bool:
        return self._target_folder.is_dir()