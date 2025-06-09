from PIL import Image
from google.cloud import storage
import io
import os
from logging import getLogger

logger = getLogger(__name__)


class DataStream:

    def __init__(self, bucket_name: str):
        """
        Args:
            bucket_name (str): GSC bucket name
        """
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(bucket_name)
        print(f"\n{__name__}:: Connected to GSC bucket: {bucket_name}")

    def get_file_list(self, file_extension: str, prefix: str) -> list:
        """
        Get a list of files in a GSC bucket with a specific prefix and file extension.

        Args:
            file_extension (str): File extension
            prefix (str): File prefix
        Returns:
            list: List of file paths
        """

        blobs = self.bucket.list_blobs(prefix=prefix)

        file_list = [
            blob.name
            for blob in blobs
            if blob.name.endswith(file_extension) and not blob.name.endswith("/")
        ]

        if not file_list:
            logger.error(
                f"No files found with extension {file_extension} and prefix {prefix}"
            )

        return file_list

    def load_image(self, file_path: str) -> Image:
        """
        Load an image from a GSC bucket.

        Args:
            file_path (str): File path
        Returns:
            Image: PIL image
        """
        try:
            blob = self.bucket.blob(file_path)
            image_bytes = blob.download_as_bytes()  # download as bytes from GSC
            image = Image.open(io.BytesIO(image_bytes))

            return image

        except Exception as e:
            logger.error(f"Failed to load image from {file_path}: {e}")

            return None

    def load_annot(self, file_path: str, file_extension: str) -> dict | list | None:
        """
        Load an annotation from a GSC bucket.

        Args:
            file_path (str): File path
            file_extension (str): File extension
        Returns:
            list | dict: Annotation
        """
        try:
            if file_extension == ".txt":
                blob = self.bucket.blob(file_path)
                annot_data = blob.download_as_text()
                annot = annot_data.split("\n")
                return annot

            if file_extension == ".json":
                # TODO: implement json loading
                pass

        except Exception as e:
            logger.error(f"Failed to load annotation from {file_path}: {e}")

            return None

    def generate_data(
        self,
        img_extension: str,
        annot_extension: str,
        img_prefix: str,
        annot_prefix: str,
        output_size: int = -1,
        return_sep: bool = False,
    ) -> list:
        """
        Generate a list of data tuples (image, annotation).

        Args:
            img_extension (str): Image file extension
            annot_extension (str): Annotation file extension
            prefix (str): File prefix
            output_size (int, optional): Output size. Defaults to -1.
        Returns:
            list: List of data tuples (image, annotation)
        """
        img_list = self.get_file_list(img_extension, img_prefix)
        annot_list = self.get_file_list(annot_extension, annot_prefix)

        paired_files = []
        images = []
        annots = []

        for img_file in img_list:
            img_basename = os.path.splitext(os.path.basename(img_file))[0]

            matched_annot = None

            for annot_file in annot_list:
                annot_basename = os.path.splitext(os.path.basename(annot_file))[0]

                if annot_basename == img_basename:
                    matched_annot = annot_file
                    break

            if matched_annot is not None:

                if return_sep:
                    images.append(img_file)
                    annots.append(matched_annot)

                else:
                    paired_files.append((img_file, matched_annot))

            else:
                logger.error(f"No matching annotation found for image {img_file}")

                return None

        if return_sep:
            if output_size != -1:
                return sorted(images[:output_size]), sorted(annots[:output_size])
            else:
                return sorted(images), sorted(annots)

        else:
            if output_size != -1:
                return paired_files[:output_size]
            else:
                return paired_files
