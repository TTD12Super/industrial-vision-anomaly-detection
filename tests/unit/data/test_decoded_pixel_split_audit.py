from PIL import Image

from scripts.audit_decoded_pixel_split_duplicates import decoded_pixel_sha256


def test_decoded_pixel_hash_ignores_png_encoding_but_detects_pixel_changes(tmp_path):
    first = tmp_path / "first.png"
    reencoded = tmp_path / "reencoded.png"
    changed = tmp_path / "changed.png"
    image = Image.new("RGB", (3, 2), (30, 80, 120))
    image.save(first, compress_level=0)
    image.save(reencoded, compress_level=9)
    image.putpixel((1, 1), (31, 80, 120))
    image.save(changed)

    assert decoded_pixel_sha256(first) == decoded_pixel_sha256(reencoded)
    assert decoded_pixel_sha256(first) != decoded_pixel_sha256(changed)
