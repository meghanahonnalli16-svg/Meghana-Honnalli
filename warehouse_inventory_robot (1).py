"""
Project 132: Warehouse Inventory Robot
=======================================
Concepts covered:
  - Python lists (SKU data structure)
  - Bucket Sort by warehouse zone
  - Binary Search for SKU lookup
  - NumPy restock quantity calculation
  - Data Analysis: slow-moving items (< 10th percentile)
"""

import random
import numpy as np
from datetime import date, timedelta

# ─────────────────────────────────────────────
# 1. DATA STRUCTURE
#    Each item is a list: [SKU, Location, Quantity, LastRestocked, PickPriority]
# ─────────────────────────────────────────────

ZONES = ["A", "B", "C", "D"]
ZONE_NAMES = {"A": "Electronics", "B": "Clothing", "C": "Household", "D": "Food & Beverage"}
NUM_SKUS = 10_000

random.seed(42)

def generate_inventory(n: int) -> list:
    """Generate n SKU records as lists."""
    inventory = []
    base_date = date(2024, 1, 1)

    for i in range(1, n + 1):
        sku          = f"SKU-{i:04d}"
        zone         = random.choice(ZONES)
        row          = random.randint(1, 20)
        col          = random.randint(1, 50)
        location     = f"{zone}{row:02d}-{col:02d}"
        quantity     = random.randint(1, 500)
        last_restock = base_date + timedelta(days=random.randint(0, 364))
        pick_priority = round(random.uniform(0.0, 1.0), 2)

        # [SKU, Location, Quantity, LastRestocked, PickPriority]
        inventory.append([sku, location, quantity, last_restock, pick_priority])

    return inventory


# ─────────────────────────────────────────────
# 2. BUCKET SORT  —  O(n + k)
#    Sort inventory into zone buckets for efficient robot routing.
# ─────────────────────────────────────────────

def bucket_sort_by_zone(inventory: list) -> dict:
    """
    Bucket-sort items by warehouse zone (A, B, C, D).
    Returns a dict { zone: [items...] } sorted by PickPriority desc within each bucket.
    """
    buckets = {zone: [] for zone in ZONES}

    for item in inventory:
        zone = item[1][0]           # first character of Location is the zone letter
        buckets[zone].append(item)

    # Sort each bucket by PickPriority (index 4) descending
    for zone in buckets:
        buckets[zone].sort(key=lambda x: x[4], reverse=True)

    return buckets


# ─────────────────────────────────────────────
# 3. BINARY SEARCH  —  O(log n)
#    Find a SKU's location during order fulfillment.
#    Requires the inventory list to be sorted by SKU first.
# ─────────────────────────────────────────────

def binary_search_sku(sorted_inventory: list, target_sku: str) -> dict:
    """
    Binary search for target_sku in a SKU-sorted inventory.
    Returns a result dict with found status, index, item, and step count.
    """
    lo, hi = 0, len(sorted_inventory) - 1
    steps = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        mid_sku = sorted_inventory[mid][0]
        steps += 1

        if mid_sku == target_sku:
            return {
                "found": True,
                "index": mid,
                "item": sorted_inventory[mid],
                "steps": steps
            }
        elif mid_sku < target_sku:
            lo = mid + 1
        else:
            hi = mid - 1

    return {"found": False, "index": -1, "item": None, "steps": steps}


# ─────────────────────────────────────────────
# 4. NUMPY — OPTIMAL RESTOCK QUANTITIES
#    restock_qty = max_capacity * (1 - turnover_rate)
#    High turnover → small top-up; low turnover → large fill.
# ─────────────────────────────────────────────

def calculate_restock_quantities(inventory: list, max_capacity: int = 500) -> np.ndarray:
    """
    Use NumPy to calculate optimal restock quantities for all SKUs.
    turnover_rate is derived from current quantity / max_capacity.
    Returns a NumPy array of restock quantities.
    """
    quantities = np.array([item[2] for item in inventory], dtype=np.float64)
    turnover_rates = quantities / max_capacity          # 0.0 (empty) → 1.0 (full)
    restock_qtys = np.round(max_capacity * (1 - turnover_rates)).astype(int)
    return restock_qtys


# ─────────────────────────────────────────────
# 5. DATA ANALYSIS — SLOW-MOVING ITEMS
#    Flag items below the 10th percentile turnover for clearance.
# ─────────────────────────────────────────────

def find_slow_movers(inventory: list, percentile: float = 10.0) -> list:
    """
    Identify slow-moving items whose turnover rate is below the given percentile.
    Returns a list of (item, turnover_rate) tuples flagged for clearance.
    """
    max_qty = max(item[2] for item in inventory)
    turnover_rates = np.array([item[2] / max_qty for item in inventory])

    cutoff = np.percentile(turnover_rates, percentile)

    slow_movers = [
        (inventory[i], round(float(turnover_rates[i]), 4))
        for i in range(len(inventory))
        if turnover_rates[i] < cutoff
    ]

    # Sort by turnover rate ascending (slowest first)
    slow_movers.sort(key=lambda x: x[1])
    return slow_movers


# ─────────────────────────────────────────────
# 6. MAIN — DEMO
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("   WAREHOUSE INVENTORY ROBOT  —  Project 132")
    print("=" * 60)

    # ── Generate inventory ──────────────────────────────────────
    print(f"\n[1] Generating {NUM_SKUS:,} SKU records...")
    inventory = generate_inventory(NUM_SKUS)
    print(f"    Sample item → {inventory[0]}")

    # ── Bucket Sort ─────────────────────────────────────────────
    print("\n[2] Bucket Sort by warehouse zone...")
    buckets = bucket_sort_by_zone(inventory)
    for zone, items in buckets.items():
        print(f"    Zone {zone} ({ZONE_NAMES[zone]}): {len(items):,} SKUs  "
              f"| Top pick: {items[0][0]} @ priority {items[0][4]}")

    # ── Binary Search ────────────────────────────────────────────
    print("\n[3] Binary Search — SKU lookup during order fulfillment...")
    sorted_inventory = sorted(inventory, key=lambda x: x[0])   # sort by SKU

    for test_sku in ["SKU-0001", "SKU-5000", "SKU-9999", "SKU-1234"]:
        result = binary_search_sku(sorted_inventory, test_sku)
        if result["found"]:
            item = result["item"]
            print(f"    FOUND  {test_sku} in {result['steps']:2d} steps → "
                  f"Location: {item[1]}, Qty: {item[2]}, Priority: {item[4]}")
        else:
            print(f"    NOT FOUND: {test_sku} ({result['steps']} steps)")

    max_steps = len(str(NUM_SKUS).encode()).bit_length()
    print(f"    (Max possible steps for {NUM_SKUS:,} items = {max_steps}  [O(log n)])")

    # ── NumPy Restock Calculation ────────────────────────────────
    print("\n[4] NumPy — Optimal restock quantities (max capacity = 500)...")
    restock = calculate_restock_quantities(inventory, max_capacity=500)
    print(f"    Total units to restock across all SKUs: {restock.sum():,}")
    print(f"    Average restock per SKU:                {restock.mean():.1f} units")
    print(f"    Max restock needed:                     {restock.max()} units")
    print(f"    SKUs that need no restock (full):        {(restock == 0).sum():,}")
    # Show first 5
    print("    First 5 SKUs and their restock amounts:")
    for i in range(5):
        print(f"      {inventory[i][0]}  qty={inventory[i][2]:3d}  → restock +{restock[i]}")

    # ── Slow Movers ──────────────────────────────────────────────
    print("\n[5] Data Analysis — Slow-moving items (< 10th percentile)...")
    slow = find_slow_movers(inventory, percentile=10.0)
    print(f"    Items flagged for clearance: {len(slow):,}")
    print("    Bottom 5 slowest movers:")
    for item, rate in slow[:5]:
        print(f"      {item[0]}  Location: {item[1]}  Qty: {item[2]}  "
              f"Turnover: {rate:.4f}  → CLEARANCE")

    print("\n" + "=" * 60)
    print("   All systems nominal. Robots ready for dispatch.")
    print("=" * 60)


if __name__ == "__main__":
    main()
