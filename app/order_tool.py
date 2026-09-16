"""
Order lookup tool. This is the ONLY way order data reaches the model -- the
full orders.json is never put in a prompt.

Rules implemented here (from data/orders-data-dictionary.md):
- Normalize harmless input differences (case, whitespace, punctuation).
- Unknown / malformed IDs -> safe error, no guessing.
- Only customer-safe fields are returned; `customer`, `internal.*` are
  stripped unconditionally, even if the caller asks for them.
- `status` is authoritative. If cancelled/returned, stale shipping/estimate
  fields are dropped so the agent can't claim the order is "still arriving".
- If status == shipped and estimated_delivery is null, we say so explicitly
  rather than letting the model invent a date.
- If status == exception, we flag `needs_human_handoff`.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any


CUSTOMER_SAFE_FIELDS = {
    "order_id",
    "membership_tier",
    "placed_at",
    "status",
    "status_updated_at",
    "shipped_at",
    "delivered_at",
    "carrier",
    "tracking_number",
    "estimated_delivery",
    "customer_safe_message",
}

ORDER_ID_RE = re.compile(r"^ORD-\d{4,}$")


@dataclass
class OrderLookupResult:
    ok: bool
    reason: str | None = None
    data: dict[str, Any] | None = None
    needs_human_handoff: bool = False


class OrderStore:
    def __init__(self, orders_path: str):
        with open(orders_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.snapshot_at: str = raw["snapshot_at"]
        self._orders: dict[str, dict[str, Any]] = {o["order_id"]: o for o in raw["orders"]}

    @staticmethod
    def normalize_order_id(raw_id: str) -> str:
        cleaned = raw_id.strip().strip(".,;:'\"").upper()
        return cleaned

    def lookup(self, raw_order_id: str) -> OrderLookupResult:
        if not raw_order_id or not raw_order_id.strip():
            return OrderLookupResult(ok=False, reason="missing_order_id")

        normalized = self.normalize_order_id(raw_order_id)

        if not ORDER_ID_RE.match(normalized):
            return OrderLookupResult(ok=False, reason="malformed_order_id")

        order = self._orders.get(normalized)
        if order is None:
            return OrderLookupResult(ok=False, reason="not_found")

        safe = {k: v for k, v in order.items() if k in CUSTOMER_SAFE_FIELDS}
        # items: keep name/quantity/final_sale only
        safe["items"] = [
            {"name": it.get("name"), "quantity": it.get("quantity"), "final_sale": it.get("final_sale")}
            for it in order.get("items", [])
        ]

        status = order.get("status")
        needs_handoff = False

        if status in ("cancelled", "returned"):
            # Drop stale forward-looking fields.
            safe["carrier"] = None
            safe["tracking_number"] = None
            safe["estimated_delivery"] = None
        elif status == "shipped" and safe.get("estimated_delivery") is None:
            safe["estimated_delivery_note"] = "unavailable"
        elif status == "exception":
            needs_handoff = True

        return OrderLookupResult(ok=True, data=safe, needs_human_handoff=needs_handoff)


if __name__ == "__main__":
    store = OrderStore(os.path.join(os.path.dirname(__file__), "..", "data", "orders.json"))
    for test_id in ["  ord-1001 ", "ORD-9999", "not-an-id", ""]:
        print(test_id, "->", store.lookup(test_id))
