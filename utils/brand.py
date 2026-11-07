"""The app's logo (assets/logo.svg), inlined so it needs no extra request and stays crisp at any size."""
import os
from functools import lru_cache

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
FAVICON = os.path.join(ASSETS, "favicon.png")


@lru_cache(maxsize=4)
def logo_svg(size: int) -> str:
    with open(os.path.join(ASSETS, "logo.svg"), encoding="utf-8") as f:
        svg = f.read().strip()
    return svg.replace("<svg ", f'<svg width="{size}" height="{size}" ', 1)
