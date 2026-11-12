"""Read development purchases without materializing held-out record fields."""
import csv
import io
from pathlib import Path
import pandas as pd
from recommender.split import TEST_START


def load_development_purchases(path):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.reader(source)
        header = next(reader)
        date_index = header.index("purchase_date")
        writer.writerow(header)
        for row in reader:
            date = pd.Timestamp(row[date_index])
            if pd.isna(date):
                raise ValueError("Purchase partitioning requires valid dates.")
            if date < TEST_START:
                writer.writerow(row)
    buffer.seek(0)
    frame = pd.read_csv(buffer, dtype={"customer_id": str, "product_id": str}, low_memory=False)
    frame["purchase_date"] = pd.to_datetime(frame["purchase_date"], format="mixed")
    return frame
