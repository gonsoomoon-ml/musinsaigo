import glob
import io
import os
import ssl
import sys
import time
from typing import Final, Optional
from urllib.error import HTTPError
from urllib.request import build_opener, install_opener, urlretrieve
import boto3
import requests
from PIL import Image, UnidentifiedImageError
from pexels_api import API
from pyunsplash import PyUnsplash
from requests.exceptions import ConnectionError, InvalidSchema, SSLError
from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    ElementNotInteractableException,
    NoSuchElementException,
)
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


ssl._create_default_https_context = ssl._create_unverified_context
EXEC_HEADLESS: Final = False
MAX_NAME_LEN: Final = 100


def download_pexels_images(
    api_key: str,
    query: str,
    num_images: int,
    images_dir: str,
    max_num_request_per_hour: int = 80,
) -> None:
    pexels = API(api_key)
    quotient, remainder = divmod(num_images, max_num_request_per_hour)
    total_downloaded = 0

    for i in tqdm(range(quotient + 1)):
        count = max_num_request_per_hour if i < quotient else remainder
        if count <= 0:
            continue

        pexels.search(query, page=i + 1, results_per_page=count)
        photos = pexels.get_entries()
        for photo in tqdm(photos, leave=False):
            start_time = time.time()
            image_name = f"pexels_{photo.id}.jpg"
            response = requests.get(photo.original, allow_redirects=True)
            with open(os.path.join(images_dir, image_name), "wb") as img_file:
                img_file.write(response.content)
            total_downloaded += 1
            logger.info("Downloaded '%d' image: '%s'.", total_downloaded, image_name)
            sleep_duration = (
                3600 // max_num_request_per_hour - round(time.time() - start_time) + 1
            )
            time.sleep(sleep_duration)


def download_unsplash_images(
    api_key: str,
    query: str,
    num_images: int,
    images_dir: str,
    featured: bool = True,
    max_num_request_per_hour: int = 50,
) -> None:
    unsplash = PyUnsplash(api_key=api_key)
    quotient, remainder = divmod(num_images, max_num_request_per_hour)
    total_downloaded = 0

    for i in tqdm(range(quotient + 1)):
        count = max_num_request_per_hour if i < quotient else remainder
        if count <= 0:
            continue

        photos = unsplash.photos(
            type_="random",
            count=count,
            query=query,
            featured=featured,
        )
        for photo in tqdm(photos.entries, leave=False):
            start_time = time.time()
            image_name = f"unsplash_{photo.id}.jpg"
            response = requests.get(photo.link_download, allow_redirects=True)
            with open(os.path.join(images_dir, image_name), "wb") as img_file:
                img_file.write(response.content)
            total_downloaded += 1
            logger.info("Downloaded '%d' image: '%s'.", total_downloaded, image_name)
            sleep_duration = (
                3600 // max_num_request_per_hour - round(time.time() - start_time) + 1
            )
            time.sleep(sleep_duration)


def download_google_images(
    query: str, num_images: int, images_dir: str, min_image_size: int = 1024
) -> None:
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    if EXEC_HEADLESS:
        options.add_argument("--headless")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )

    url = f"https://www.google.com/search?q={query}&source=lnms&tbm=isch"
    driver.get(url)

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        try:
            button = driver.find_element(
                By.XPATH, '//*[@id="islmp"]/div/div/div/div/div[1]/div[2]/div[2]/input'
            )
            if button.is_displayed():
                button.click()
        except ElementNotInteractableException:
            pass
        time.sleep(1)
        if (
            driver.find_element(By.CLASS_NAME, "OuJzKb.Yu2Dnd").text
            == "더 이상 표시할 콘텐츠가 없습니다."
        ):
            break

    images = driver.find_elements(By.CSS_SELECTOR, ".rg_i.Q4LuWd")
    logger.info("There are a total of %d images in the Google search.", len(images))
    total_downloaded = 0

    for image in images:
        try:
            image.click()
            time.sleep(1)
            image_url = driver.find_element(
                By.XPATH,
                '//*[@id="Sva75c"]/div[2]/div[2]/div[2]/div[2]/c-wiz/div/div/div/div/div[3]/div[1]/a/img[1]',
            ).get_attribute("src")
            opener = build_opener()
            opener.addheaders = [
                (
                    "User-Agent",
                    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36",
                )
            ]
            install_opener(opener)

            response = requests.get(image_url, stream=True)
            image = Image.open(io.BytesIO(response.content))
            image_width, image_height = image.size
            if image_width >= min_image_size or image_height >= min_image_size:
                total_downloaded += 1
                image_name = f"google_{image_url.split('/')[-1].split('.')[0].split('?')[0][:MAX_NAME_LEN]}.jpg"
                with open(os.path.join(images_dir, image_name), "wb") as file:
                    file.write(response.content)
                msg = f"The '{total_downloaded}' image '{image_name}' has been downloaded."
                logger.info(msg)

        except (
            ConnectionError,
            ElementClickInterceptedException,
            ElementNotInteractableException,
            HTTPError,
            InvalidSchema,
            NoSuchElementException,
            SSLError,
            UnidentifiedImageError,
        ) as error:
            msg = f"An error occurred while downloading the image. ({error})"
            logger.warning(msg)

        if total_downloaded > num_images:
            break

    driver.close()


def download_musinsa_images(
    num_images: int,
    images_dir: str,
    is_street_snap: bool,
    order_method: Optional[str],
    is_best: bool,
    num_images_per_page: int = 60,
) -> None:
    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    if EXEC_HEADLESS:
        options.add_argument("--headless")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )

    quotient, remainder = divmod(num_images, num_images_per_page)
    total_downloaded = 0
    for i in tqdm(range(quotient + 1)):
        url = (
            f"https://www.musinsa.com/mz/streetsnap?{get_musinsa_filter(order_method, is_best)}p={i + 1}"
            if is_street_snap
            else f"https://www.musinsa.com/mz/brandsnap?{get_musinsa_filter(order_method, is_best)}p={i + 1}"
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

                    image_name = f"musinsa_{image_url.split('/')[-1].split('.')[1].split('?')[-1]}.jpg"
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

                except (HTTPError, IndexError) as error:
                    msg = f"There was an error downloading the '{j + 1}' image on page '{i + 1}'. ({error})"
                    logger.warning(msg)

                driver.get(url)


def get_musinsa_filter(order_method: Optional[str] = None, is_best: bool = True) -> str:
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

    if config.data_source.lower() == "pexels":
        download_pexels_images(
            config.pexels_api_key,
            config.data_query,
            config.num_images,
            images_dir,
        )
    elif config.data_source.lower() == "unsplash":
        download_unsplash_images(
            config.unsplash_api_key,
            config.data_query,
            config.num_images,
            images_dir,
        )
    elif config.data_source.lower() == "musinsa":
        download_musinsa_images(
            config.num_images,
            images_dir,
            config.is_street_snap,
            config.order_method,
            config.is_best,
        )
    else:
        download_google_images(
            config.data_query,
            config.num_images,
            images_dir,
        )

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
