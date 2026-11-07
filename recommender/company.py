"""Structure and load companies metadata for further training."""

from pathlib import Path
from html import unescape
import re
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recommender.data import load_purchases
from recommender.merging import merge_purchase_events, merge_records
from recommender.scope import DEFAULT_DATASETS_PATH, actual_purchase_mask, get_datasets_locs

COMPANY_COLUMN_MAPPING = {
    1: "customer_id",
    2: "company_name",
    82: "legal_name",
    85: "region_code",
    88: "address",
    90: "description",
    94: "industry_branch",
}

EXCLUDED_COMPANY_COLUMNS = {
    44: "sales_branch",
    59: "vat_status",
    98: "city_code",
}

BASE_COMPANY_COLUMNS = {1, 2, 82, 88, 90}

LOCALITY_PREFIX_RE = re.compile(
    r"(?<!\w)(?:(?:смт|пгт|пос|м|г|с|c)(?:\.|\s+)"
    r"|(?:місто|город|село|селище|с-ще|пос[её]лок)\s+)", re.IGNORECASE,
)
ADDRESS_DETAIL_RE = re.compile(
    r"(?<!\w)(?:вул(?:иця)?|ул(?:ица)?|просп(?:ект)?|пр|пр-т|просп-т"
    r"|пров(?:улок)?|пер(?:еулок)?|бул(?:ьвар)?|бульвар|б-р|пл(?:оща|ощадь)?"
    r"|шосе|шоссе|проїзд|проезд|узвіз|набережна|наб(?:ережная)?"
    r"|буд(?:инок)?|дом|д|б|корп(?:ус)?|кв(?:артира)?|офіс|офис"
    r"|відділення|отделение|поштомат|ринок|рынок|р-к|мікрорайон|микрорайон"
    r"|мкр|територія|территория|тер-я|гараж|бокс|ГСК|ГК"
    r"|street|st|road|rd|avenue|ave|lane|boulevard|building|apartment)(?:\.|\b)",
    re.IGNORECASE,
)
ADMIN_RE = re.compile(
    r"(?<!\w)(?:область|обл|район(?:у|а)?|р-н(?:у|а)?|р-он|АР\s+Кр[иы]м)(?:\.|\b)",
    re.IGNORECASE,
)
# Include the preceding adjective when trimming an inline administrative area,
# e.g. "Кривий Ріг Дніпропетровська обл." -> "Кривий Ріг".
ADMIN_PHRASE_RE = re.compile(
    r"(?<!\w)(?:АР\s+Кр[иы]м|(?:[^\W\d_]+(?:-[^\W\d_]+)*\s+)?"
    r"(?:область|обл|район(?:у|а)?|р-н(?:у|а)?|р-он))(?:\.|\b)",
    re.IGNORECASE,
)
COUNTRY_RE = re.compile(
    r"\b(?:Україна|Украина|Ukraine|UA|Росія|Россия|Russia|Німеччина|Германия"
    r"|Germany|Польща|Польша|Poland|Білорусь|Беларусь|Belarus|Молдова|Moldova)\b",
    re.IGNORECASE,
)
LOCALITY_NAME_RE = re.compile(
    r"[^\W\d_]+(?:[-'’ʼ][^\W\d_]+)*(?:-\d+)?"
    r"(?:\s+[^\W\d_]+(?:[-'’ʼ][^\W\d_]+)*(?:-\d+)?)*",
)
HOUSE_NUMBER_RE = re.compile(r"^\d{1,4}(?:\D|$)")


def _is_locality_name(value: str) -> bool:
    return (
        len(value) > 1 and LOCALITY_NAME_RE.fullmatch(value) is not None
        and all(len(word) > 1 for word in value.split())
        and not any(pattern.search(value) for pattern in (ADDRESS_DETAIL_RE, ADMIN_RE, COUNTRY_RE))
    )


def process_address(address: object) -> str | None:
    """Extract a locality from an address; return None for missing/ambiguous data.

    Prefer explicit Ukrainian/Russian locality markers, trimming inline street,
    number and administrative details. Without markers, exclude address details
    and bare street names followed by house numbers. Prefer the remaining name
    closest before an administrative area, otherwise require one unique name.
    Preserve source spelling; do not translate or guess conflicting localities.
    """
    if not isinstance(address, str) or not address.strip() or address.strip().upper() == "NULL":
        return None
    address = re.sub(r"[;\r\n]+", ",", unescape(address))
    address = re.sub(r"\s+", " ", address).strip()
    parts = [part.strip(" .:") for part in address.split(",") if part.strip(" .:")]

    marked = {}
    has_locality_marker = False
    for part in parts:
        markers = list(LOCALITY_PREFIX_RE.finditer(part))
        for index, marker in enumerate(markers):
            prefix = part[:marker.start()]
            if ADDRESS_DETAIL_RE.search(prefix) and not re.search(r"\d", prefix):
                continue
            has_locality_marker = True
            end = markers[index + 1].start() if index + 1 < len(markers) else len(part)
            name = part[marker.end():end]
            boundaries = [pattern.search(name) for pattern in (
                ADDRESS_DETAIL_RE, ADMIN_PHRASE_RE, COUNTRY_RE,
            )]
            number = re.search(r"(?<![\w-])\d", name)
            boundaries.append(number)
            end = min((match.start() for match in boundaries if match), default=len(name))
            name = name[:end].strip(" .:-")
            if number and end == number.start() and len(name.split()) > 1:
                if not re.match(r"\d{5,6}(?:\D|$)", number.string[number.start():]):
                    continue
            if _is_locality_name(name):
                marked.setdefault(name.casefold(), name)
    if has_locality_marker:
        return next(iter(marked.values())) if len(marked) == 1 else None

    candidates = []
    for index, part in enumerate(parts):
        name = re.sub(r"^\d{5,6}\s+", "", part)
        if not _is_locality_name(name):
            continue
        if index + 1 < len(parts) and HOUSE_NUMBER_RE.match(parts[index + 1]):
            continue
        candidates.append((index, name))
    for index, part in enumerate(parts):
        if ADMIN_RE.search(part):
            before = [name for position, name in candidates if position < index]
            if before:
                return before[-1]
    unique = {name.casefold(): name for _, name in candidates}
    return next(iter(unique.values())) if len(unique) == 1 else None


def count_company_purchases(purchases: pd.DataFrame) -> pd.Series:
    """Count distinct valid paid date/product events per company.

    Accept the normalized schema returned by ``load_purchases``. Duplicate lines
    for the same company, calendar date and product count as one event.
    Gift, return, invalid and nonmerchandise records never count as purchases.
    """
    paid = merge_purchase_events(purchases.loc[actual_purchase_mask(purchases)])
    return paid.groupby("customer_id").size().rename("purchase_count")


def load_companies(
    input_path: str | Path | None = None,
    *, config_path: str | Path = DEFAULT_DATASETS_PATH,
    purchases: pd.DataFrame | None = None,
    minimum_purchases: int = 1,
) -> pd.DataFrame:
    """Load metadata and keep companies meeting the paid-purchase threshold.

    Default to at least one valid purchase event. Set minimum_purchases=0
    to retain every company, including those with no purchases. Input paths are
    resolved from TOML when omitted; supplied purchases reuse prepared history.
    """
    if type(minimum_purchases) is not int or minimum_purchases < 0:
        raise ValueError("minimum_purchases must be a nonnegative integer.")
    if input_path is None:
        input_path = get_datasets_locs("companies", config_path)

    companies_raw = pd.read_csv(
        input_path, sep=";", header=None, dtype="string",
        encoding="utf-8-sig", keep_default_na=False,
    )
    if not BASE_COMPANY_COLUMNS.issubset(companies_raw.columns):
        missing = sorted(set(BASE_COMPANY_COLUMNS) - set(companies_raw.columns))
        raise ValueError(f"Company export is missing required zero-based columns: {missing}")

    has_extended = {85, 94}.issubset(companies_raw.columns)
    available_indices = [
        c for c in COMPANY_COLUMN_MAPPING
        if c not in EXCLUDED_COMPANY_COLUMNS and c in companies_raw.columns
    ]
    if not has_extended:
        available_indices = [c for c in available_indices if c in BASE_COMPANY_COLUMNS]

    mapping = {c: COMPANY_COLUMN_MAPPING[c] for c in available_indices}
    companies = companies_raw[available_indices].rename(columns=mapping).copy()

    for column in companies.columns:
        companies[column] = companies[column].str.strip()
        companies[column] = companies[column].mask(companies[column].isin(["", "NULL"]))
    if companies["customer_id"].isna().any():
        raise ValueError("Company export contains missing customer IDs.")
    companies = merge_records(companies, keys=["customer_id"])
    companies["city"] = companies["address"].map(process_address).astype("string")
    if purchases is None:
        purchases = load_purchases(
            get_datasets_locs("purchases", config_path),
            items_path=get_datasets_locs("items", config_path),
        )
    counts = count_company_purchases(purchases)
    companies["purchase_count"] = companies["customer_id"].map(counts).fillna(0).astype("int64")

    if has_extended:
        ordered_cols = [
            "customer_id", "company_name", "legal_name", "address",
            "description", "city", "region_code", "industry_branch", "purchase_count",
        ]
        companies = companies[[c for c in ordered_cols if c in companies.columns]]
    else:
        ordered_cols = [
            "customer_id", "company_name", "legal_name", "address",
            "description", "city", "purchase_count",
        ]
        companies = companies[[c for c in ordered_cols if c in companies.columns]]

    return companies.loc[companies["purchase_count"].ge(minimum_purchases)].reset_index(drop=True)


if __name__ == "__main__":
    companies = load_companies()
    print(f"companies: {len(companies)} rows; city extracted: {companies.city.notna().sum()}")
