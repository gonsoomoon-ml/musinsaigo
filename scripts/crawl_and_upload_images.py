import glob
import os
import sys
import time
from typing import Final, Optional
from urllib.error import HTTPError
from urllib.request import urlretrieve
import boto3
import requests
from pyunsplash import PyUnsplash
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from tqdm import tqdm
from webdriver_manager.chrome import ChromeDriverManager

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), os.pardir))
)
from utils.enums import DirName, FileName
from utils.config_handler import load_config
from utils.logger import logger
from utils.misc import create_bucket_if_not_exists, upload_dir_to_s3


MAX_NUM_REQUEST_PER_HOUR: Final = 50
NUM_IMAGES_PER_PAGE: Final = 60


def download_unsplash_images(
    data_source: str,
    api_key: str,
    query: str,
    num_images: int,
    images_dir: str,
    max_num_request_per_hour: int,
) -> None:
    unsplash = PyUnsplash(api_key=api_key)
    quotient, remainder = divmod(num_images, max_num_request_per_hour)
    total_downloaded = 0

    for _ in tqdm(range(quotient + 1)):
        count = max_num_request_per_hour if _ < quotient else remainder
        if count <= 0:
            continue

        photos = unsplash.photos(
            type_="random", count=count, featured=True, query=query
        )
        for photo in tqdm(photos.entries, leave=False):
            start_time = time.time()
            image_name = f"{data_source}_{photo.id}.jpg"
            response = requests.get(photo.link_download, allow_redirects=True)
            with open(os.path.join(images_dir, image_name), "wb") as img_file:
                img_file.write(response.content)
            total_downloaded += 1
            logger.info("Downloaded '%d' image: '%s'.", total_downloaded, image_name)
            sleep_duration = (
                3600 // max_num_request_per_hour - round(time.time() - start_time) + 1
            )
            time.sleep(sleep_duration)


def download_musinsa_images(
    data_source: str,
    num_images: int,
    images_dir: str,
    num_images_per_page: int,
    is_street_snap: bool,
    order_method: Optional[str],
    is_best: bool,
) -> None:
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1420,1080")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )

    quotient, remainder = divmod(num_images, num_images_per_page)
    total_downloaded = 0
    for i in tqdm(range(quotient + 1)):
        url = (
            f"https://www.musinsa.com/mz/streetsnap?{get_filter(order_method, is_best)}p={i + 1}"
            if is_street_snap
            else f"https://www.musinsa.com/mz/brandsnap?{get_filter(order_method, is_best)}p={i + 1}"
        )
        driver.get(url)

        count = num_images_per_page if i < quotient else remainder
        if count > 0:
            for j in tqdm(range(count), leave=False):
                try:
                    driver.find_elements(by=By.CSS_SELECTOR, value=".articleImg")[
                        j
                    ].click()
                    image_url = (
                        driver.find_elements(by=By.CSS_SELECTOR, value=".lbox")[
                            0
                        ].get_attribute("href")
                        if is_street_snap
                        else driver.find_element(
                            by=By.XPATH, value="//meta[@property='og:image']"
                        ).get_attribute("content")
                    )

                    image_name = f"{data_source}_{image_url.split('/')[-1].split('.')[1].split('?')[-1]}.jpg"
                    urlretrieve(
                        image_url,
                        os.path.join(
                            images_dir,
                            image_name,
                        ),
                    )

                    total_downloaded += 1
                    msg = f"The '{total_downloaded}' image (the '{j + 1}' image on page '{i + 1}'), '{image_name}' has been downloaded."
                    logger.info(msg)

                except (HTTPError, IndexError):
                    msg = f"There was an error downloading the '{j + 1}' image on page '{i + 1}'."
                    logger.warning(msg)

                driver.get(url)


def get_filter(order_method: Optional[str] = None, is_best: bool = True) -> str:
    if order_method == "best":
        order_method_str = "ordw=best&"
    elif order_method == "hit":
        order_method_str = "ordw=hit&"
    elif order_method == "comment":
        order_method_str = "ordw=comment&"
    elif order_method == "inc":
        order_method_str = "ordw=inc&"
    elif order_method == "d_comment":
        order_method_str = "ordw=d_comment&"
    else:
        order_method_str = ""

    is_best_str = "bst=1&" if is_best else ""

    return order_method_str + is_best_str


if __name__ == "__main__":
    logger.info("The image crawling task started...")

    config_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            DirName.CONFIGS.value,
            FileName.CONFIG.value,
        )
    )
    config = load_config(config_path)

    images_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, config.images_prefix)
    )
    images_prefix = (
        f"{config.base_prefix}/raw_{config.dataset_prefix}/{config.images_prefix}"
    )

    boto_session = boto3.Session(
        profile_name=config.profile_name, region_name=config.region_name
    )
    os.makedirs(images_dir, exist_ok=True)

    image_paths = glob.glob(os.path.join(images_dir, "*.jpg"))
    logger.info("Starting with %s images...", len(image_paths))

    if config.data_source.lower() == "unsplash":
        download_unsplash_images(
            config.data_source,
            config.unsplash_api_key,
            config.data_query,
            config.num_images,
            images_dir,
            MAX_NUM_REQUEST_PER_HOUR,
        )
    elif config.data_source.lower() == "musinsa":
        download_musinsa_images(
            config.data_source,
            config.num_images,
            images_dir,
            NUM_IMAGES_PER_PAGE,
            config.is_street_snap,
            config.order_method,
            config.is_best,
        )
    else:
        raise ValueError("Supported crawl targets are 'unsplash' or 'musinsa'.")

    bucket = (
        config.bucket
        if config.bucket
        else create_bucket_if_not_exists(
            boto_session, config.region_name, logger=logger
        )
    )

    upload_dir_to_s3(
        boto_session,
        images_dir,
        bucket,
        images_prefix,
        file_ext_to_excl=["DS_Store"],
        logger=logger,
    )

    logger.info("The image crawling task ended successfully.")
