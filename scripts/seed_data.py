"""
Seed the DuckDB sample database with schema + realistic data.

Run this once before starting the project:
    python scripts/seed_data.py

What this creates (data/sample_db.duckdb):
    customers    — 30 rows across 6 countries
    products     — 20 rows across 4 categories
    orders       — 60 rows with various statuses and dates
    order_items  — ~150 rows linking orders to products

Why DuckDB instead of SQLite for query execution?
    - DuckDB has richer SQL support (DATE_TRUNC, QUALIFY, window functions, etc.)
      which mirrors what you'd see in Snowflake
    - It's the "local Snowflake substitute" in this project
    - Runs in-process, zero server setup
"""

import sys
from pathlib import Path

# Allow running from repo root: python scripts/seed_data.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
from app.core.config import settings


def seed() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    db_path = str(settings.duckdb_path)
    print(f"Creating DuckDB at: {db_path}")

    con = duckdb.connect(db_path)

    # ── Schema ───────────────────────────────────────────────────────────────
    # DuckDB uses execute(), not executescript() — run statements individually.
    for stmt in [
        "DROP TABLE IF EXISTS order_items",
        "DROP TABLE IF EXISTS orders",
        "DROP TABLE IF EXISTS products",
        "DROP TABLE IF EXISTS customers",
        """CREATE TABLE customers (
            customer_id  INTEGER PRIMARY KEY,
            name         VARCHAR NOT NULL,
            email        VARCHAR NOT NULL,
            country      VARCHAR NOT NULL,
            created_at   DATE    NOT NULL
        )""",
        """CREATE TABLE products (
            product_id  INTEGER PRIMARY KEY,
            name        VARCHAR NOT NULL,
            category    VARCHAR NOT NULL,
            price       DECIMAL(10, 2) NOT NULL
        )""",
        """CREATE TABLE orders (
            order_id      INTEGER PRIMARY KEY,
            customer_id   INTEGER NOT NULL REFERENCES customers(customer_id),
            order_date    DATE    NOT NULL,
            total_amount  DECIMAL(10, 2) NOT NULL,
            status        VARCHAR NOT NULL
        )""",
        """CREATE TABLE order_items (
            item_id     INTEGER PRIMARY KEY,
            order_id    INTEGER NOT NULL REFERENCES orders(order_id),
            product_id  INTEGER NOT NULL REFERENCES products(product_id),
            quantity    INTEGER NOT NULL,
            unit_price  DECIMAL(10, 2) NOT NULL
        )""",
    ]:
        con.execute(stmt)

    # ── Customers ─────────────────────────────────────────────────────────────
    customers = [
        (1,  "Alice Müller",    "alice@email.de",    "Germany",       "2022-03-15"),
        (2,  "Bob Smith",       "bob@email.co.uk",   "UK",            "2022-05-20"),
        (3,  "Carlos García",   "carlos@email.es",   "Spain",         "2022-07-10"),
        (4,  "Diana Chen",      "diana@email.cn",    "China",         "2022-08-01"),
        (5,  "Eva Fischer",     "eva@email.de",      "Germany",       "2022-09-12"),
        (6,  "Frank Brown",     "frank@email.us",    "USA",           "2022-10-03"),
        (7,  "Grace Kim",       "grace@email.kr",    "South Korea",   "2022-11-22"),
        (8,  "Hans Weber",      "hans@email.de",     "Germany",       "2023-01-08"),
        (9,  "Isabel López",    "isabel@email.es",   "Spain",         "2023-02-14"),
        (10, "Jack Wilson",     "jack@email.us",     "USA",           "2023-03-05"),
        (11, "Karen Lee",       "karen@email.us",    "USA",           "2023-04-17"),
        (12, "Liam Taylor",     "liam@email.co.uk",  "UK",            "2023-05-30"),
        (13, "Maria Rossi",     "maria@email.it",    "Italy",         "2023-06-11"),
        (14, "Niklas Berg",     "niklas@email.de",   "Germany",       "2023-07-22"),
        (15, "Olivia Jones",    "olivia@email.co.uk","UK",            "2023-08-09"),
        (16, "Pablo Martínez",  "pablo@email.es",    "Spain",         "2023-09-18"),
        (17, "Quinn Murphy",    "quinn@email.us",    "USA",           "2023-10-27"),
        (18, "Rosa Ferri",      "rosa@email.it",     "Italy",         "2023-11-14"),
        (19, "Stefan Braun",    "stefan@email.de",   "Germany",       "2023-12-03"),
        (20, "Tina Zhang",      "tina@email.cn",     "China",         "2024-01-19"),
        (21, "Uwe Richter",     "uwe@email.de",      "Germany",       "2024-02-07"),
        (22, "Victoria Adams",  "vic@email.co.uk",   "UK",            "2024-03-14"),
        (23, "Wei Liu",         "wei@email.cn",      "China",         "2024-04-02"),
        (24, "Xavier Dubois",   "xavier@email.fr",   "France",        "2024-05-21"),
        (25, "Yuki Tanaka",     "yuki@email.jp",     "Japan",         "2024-06-08"),
        (26, "Zara Patel",      "zara@email.in",     "India",         "2024-07-15"),
        (27, "Aaron Clark",     "aaron@email.us",    "USA",           "2024-08-03"),
        (28, "Beatriz Silva",   "bea@email.br",      "Brazil",        "2024-09-12"),
        (29, "Chris Evans",     "chris@email.us",    "USA",           "2024-10-01"),
        (30, "Dora Novak",      "dora@email.cz",     "Czech Republic","2024-10-29"),
    ]
    con.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?)", customers
    )

    # ── Products ──────────────────────────────────────────────────────────────
    products = [
        (1,  "Laptop Pro 15",      "Electronics",  1299.99),
        (2,  "Wireless Mouse",     "Electronics",    29.99),
        (3,  "USB-C Hub",          "Electronics",    49.99),
        (4,  "Mechanical Keyboard","Electronics",   109.99),
        (5,  "4K Monitor",         "Electronics",   399.99),
        (6,  "Python Cookbook",    "Books",          39.99),
        (7,  "Clean Code",         "Books",          34.99),
        (8,  "Designing Data",     "Books",          44.99),
        (9,  "SQL Mastery",        "Books",          29.99),
        (10, "The Pragmatic Prog", "Books",          49.99),
        (11, "Standing Desk",      "Furniture",     349.99),
        (12, "Ergonomic Chair",    "Furniture",     499.99),
        (13, "Desk Lamp",          "Furniture",      59.99),
        (14, "Monitor Stand",      "Furniture",      79.99),
        (15, "Cable Organizer",    "Furniture",      19.99),
        (16, "Blue T-Shirt",       "Clothing",       24.99),
        (17, "Hoodie",             "Clothing",       59.99),
        (18, "Running Shoes",      "Clothing",       89.99),
        (19, "Backpack",           "Clothing",       79.99),
        (20, "Sunglasses",         "Clothing",       49.99),
    ]
    con.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?)", products
    )

    # ── Orders ────────────────────────────────────────────────────────────────
    orders = [
        (1,  1,  "2023-01-10", 1329.98, "completed"),
        (2,  2,  "2023-01-15",   79.98, "completed"),
        (3,  3,  "2023-02-03",  409.99, "completed"),
        (4,  4,  "2023-02-20",  349.99, "completed"),
        (5,  5,  "2023-03-05",  139.98, "completed"),
        (6,  1,  "2023-03-22",   49.99, "completed"),
        (7,  6,  "2023-04-10",  499.99, "completed"),
        (8,  7,  "2023-04-25",  159.98, "completed"),
        (9,  2,  "2023-05-08",  399.99, "completed"),
        (10, 8,  "2023-05-19", 1299.99, "completed"),
        (11, 9,  "2023-06-01",   74.98, "completed"),
        (12, 3,  "2023-06-14",  189.98, "completed"),
        (13, 10, "2023-07-02",   89.99, "pending"),
        (14, 11, "2023-07-18",  579.98, "completed"),
        (15, 4,  "2023-08-05",  399.99, "completed"),
        (16, 12, "2023-08-22",  109.99, "completed"),
        (17, 13, "2023-09-09",  349.99, "cancelled"),
        (18, 5,  "2023-09-26",   59.99, "completed"),
        (19, 14, "2023-10-11",  849.98, "completed"),
        (20, 6,  "2023-10-28", 1299.99, "completed"),
        (21, 15, "2023-11-04",  149.98, "pending"),
        (22, 7,  "2023-11-21",  499.99, "completed"),
        (23, 16, "2023-12-03",   84.98, "completed"),
        (24, 1,  "2023-12-18",  399.99, "completed"),
        (25, 17, "2024-01-06",   79.99, "completed"),
        (26, 8,  "2024-01-23",  189.98, "completed"),
        (27, 18, "2024-02-07",  449.98, "cancelled"),
        (28, 9,  "2024-02-21",   64.98, "pending"),
        (29, 19, "2024-03-05", 1349.98, "completed"),
        (30, 10, "2024-03-19",  499.99, "completed"),
        (31, 20, "2024-04-02",   74.98, "completed"),
        (32, 2,  "2024-04-15",  409.99, "completed"),
        (33, 21, "2024-05-01",   99.98, "completed"),
        (34, 11, "2024-05-14",  349.99, "pending"),
        (35, 22, "2024-05-28",  169.98, "completed"),
        (36, 3,  "2024-06-10",  579.98, "completed"),
        (37, 23, "2024-06-24",   59.99, "completed"),
        (38, 12, "2024-07-08",  399.99, "completed"),
        (39, 24, "2024-07-22",  149.98, "completed"),
        (40, 4,  "2024-08-05", 1299.99, "completed"),
        (41, 25, "2024-08-19",   89.98, "cancelled"),
        (42, 13, "2024-09-02",  499.99, "completed"),
        (43, 14, "2024-09-16",  109.98, "completed"),
        (44, 5,  "2024-09-30",  799.98, "completed"),
        (45, 26, "2024-10-07",   49.99, "pending"),
        (46, 15, "2024-10-14",  339.98, "completed"),
        (47, 27, "2024-10-21",  179.98, "completed"),
        (48, 6,  "2024-11-01",  399.99, "completed"),
        (49, 28, "2024-11-11",   74.98, "completed"),
        (50, 16, "2024-11-18",  499.99, "completed"),
        (51, 29, "2024-11-25",  129.98, "completed"),
        (52, 17, "2024-12-02",  349.99, "pending"),
        (53, 7,  "2024-12-09",   89.99, "completed"),
        (54, 18, "2024-12-16",  199.98, "completed"),
        (55, 30, "2024-12-20",   59.99, "completed"),
        (56, 1,  "2024-12-22",  449.98, "completed"),
        (57, 19, "2024-12-26", 1299.99, "cancelled"),
        (58, 20, "2024-12-28",   79.98, "completed"),
        (59, 8,  "2025-01-03",  399.99, "completed"),
        (60, 21, "2025-01-10",  139.98, "pending"),
    ]
    con.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders
    )

    # ── Order Items ───────────────────────────────────────────────────────────
    order_items = [
        # order 1: laptop + mouse
        (1,  1,  1, 1, 1299.99), (2,  1,  2, 1, 29.99),
        # order 2: 2 books
        (3,  2,  6, 1, 39.99),   (4,  2,  7, 1, 39.99),
        # order 3: monitor
        (5,  3,  5, 1, 399.99),  (6,  3,  2, 1, 10.00),
        # order 4: standing desk
        (7,  4, 11, 1, 349.99),
        # order 5: keyboard + mouse
        (8,  5,  4, 1, 109.99),  (9,  5,  2, 1, 29.99),
        # order 6: usb hub
        (10, 6,  3, 1, 49.99),
        # order 7: chair
        (11, 7, 12, 1, 499.99),
        # order 8: shoes + hoodie
        (12, 8, 18, 1, 89.99),   (13, 8, 17, 1, 59.99),
        # order 9: monitor
        (14, 9,  5, 1, 399.99),
        # order 10: laptop
        (15,10,  1, 1, 1299.99),
        # order 11: 2 books
        (16,11,  9, 1, 29.99),   (17,11,  8, 1, 44.99),
        # order 12: hoodie + backpack
        (18,12, 17, 1, 59.99),   (19,12, 19, 1, 89.99),(20,12,16,1,39.99),
        # order 13: running shoes
        (21,13, 18, 1, 89.99),
        # order 14: chair + lamp
        (22,14, 12, 1, 499.99),  (23,14, 13, 1, 59.99),(24,14,16,1,20.00),
        # order 15: monitor
        (25,15,  5, 1, 399.99),
        # order 16: keyboard
        (26,16,  4, 1, 109.99),
        # order 17: standing desk (cancelled)
        (27,17, 11, 1, 349.99),
        # order 18: lamp
        (28,18, 13, 1, 59.99),
        # order 19: laptop + keyboard
        (29,19,  1, 1, 1299.99), (30,19,  4, 1, 109.99),(31,19,20,1,49.99),
        # order 20: laptop
        (32,20,  1, 1, 1299.99),
        # order 21: books
        (33,21,  6, 1, 39.99),   (34,21, 10, 1, 49.99),(35,21,9,1,29.99),
        # order 22: chair
        (36,22, 12, 1, 499.99),
        # order 23: 2 books
        (37,23,  7, 1, 34.99),   (38,23,  6, 1, 49.99),
        # order 24: monitor
        (39,24,  5, 1, 399.99),
        # order 25: backpack
        (40,25, 19, 1, 79.99),
        # order 26: hoodie + sunglasses
        (41,26, 17, 1, 59.99),   (42,26, 20, 1, 49.99),(43,26,16,1,79.99),
        # order 27: chair + lamp (cancelled)
        (44,27, 12, 1, 399.99),  (45,27, 13, 1, 49.99),
        # order 28: 2 books
        (46,28,  8, 1, 34.99),   (47,28,  9, 1, 29.99),
        # order 29: laptop + hub
        (48,29,  1, 1, 1299.99), (49,29,  3, 1, 49.99),
        # order 30: chair
        (50,30, 12, 1, 499.99),
    ]
    con.executemany(
        "INSERT INTO order_items VALUES (?, ?, ?, ?, ?)", order_items
    )

    con.close()

    total_customers = 30
    total_products  = 20
    total_orders    = 60
    total_items     = len(order_items)
    print(f"Seeded: {total_customers} customers, {total_products} products, "
          f"{total_orders} orders, {total_items} order_items")
    print("Done.")


if __name__ == "__main__":
    seed()
