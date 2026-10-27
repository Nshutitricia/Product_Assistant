import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_ECHO = 30


def _load_products() -> dict[str, dict]:
    path = Path(__file__).with_name("products.json")
    with open(path, encoding="utf-8") as f:
        return {p["id"]: p for p in json.load(f)}


PRODUCTS = _load_products()


def _format_price(price: int) -> str:
    return f"{price:,} RWF"


def search_product(query: str) -> str:
    
    if not isinstance(query, str) or not query.strip():
        return "Please provide a search term."
    needle = query.strip().lower()
    lines = [
        f"{p['id']} | {p['name']} | {p['description']} | {_format_price(p['price'])}"
        for p in PRODUCTS.values()
        if needle in p["name"].lower() or needle in p["description"].lower()
    ]
    logger.info("search_product matches=%d", len(lines))
    if not lines:
        return f"No products found matching '{query.strip()[:MAX_ECHO]}'."
    return "\n".join(lines)


def check_price(product_id: str) -> str:
    
    pid = product_id.strip().upper() if isinstance(product_id, str) else None
    product = PRODUCTS.get(pid) if pid else None
    if product is None:
        shown = str(product_id).strip()[:MAX_ECHO]
        logger.warning("check_price result=not_found product_id=%r", shown)
        return f"Product '{shown}' not found in the catalogue."
    logger.info("check_price result=found product_id=%s", pid)
    return f"{pid}: {product['name']} costs {_format_price(product['price'])}"
