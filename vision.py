"""Images in chat: validate and shrink them before a vision model reads them.

A CPU spends time per image patch, so pictures are scaled down to at most VISION_MAX_SIDE pixels
(screenshots stay readable) and re-encoded as JPEG. Anything that is not a real image is refused;
nothing is written to disk.
"""
import base64
import binascii
import io

import config

MAX_IMAGE_BYTES = 12 * 1024 * 1024


def prepare_image(data: str) -> str:
    """base64 (or a data: URL) of an image -> base64 JPEG no larger than VISION_MAX_SIDE. ValueError if bad."""
    from PIL import Image, UnidentifiedImageError

    raw_b64 = data.split(",", 1)[1] if data.startswith("data:") else data
    try:
        raw = base64.b64decode(raw_b64, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("That image could not be read (bad encoding).") from None
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("That image is over 12 MB.")
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            if img.width * img.height > 60_000_000:
                raise ValueError("That image is too large (over 60 megapixels).")
            img = img.convert("RGB")
            img.thumbnail((config.VISION_MAX_SIDE, config.VISION_MAX_SIDE))
            out = io.BytesIO()
            img.save(out, "JPEG", quality=88)
    except (UnidentifiedImageError, OSError):
        raise ValueError("That file is not an image this app can read (use PNG, JPEG, WebP, GIF or BMP).") from None
    return base64.b64encode(out.getvalue()).decode("ascii")
