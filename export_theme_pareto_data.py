import argparse
import csv
import os
import sqlite3
import zipfile
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Iterable, Optional


HEADERS = [
    "rank",
    "theme",
    "items_in_theme",
    "share_percent",
    "cumulative_items",
    "cumulative_share_percent",
    "post_items",
    "comment_items",
    "total_clustered_items",
]


def clean_label(label: Optional[str]) -> str:
    if not label:
        return "Unclustered"
    cleaned = "".join(" " if ord(char) < 32 or 127 <= ord(char) <= 159 else char for char in str(label))
    cleaned = " ".join(cleaned.split())
    return cleaned or "Unclustered"


def find_database_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.glob("*.db")
        if path.name not in {"test.db"} and path.is_file()
    )


def choose_database(default_dir: Path) -> Path:
    db_files = find_database_files(default_dir)
    if not db_files:
        raise FileNotFoundError(f"No .db files found in {default_dir}")

    print("Available workspace databases:")
    for idx, path in enumerate(db_files, 1):
        print(f"  {idx}. {path.name}")

    choice = input("Choose database number or enter a DB path: ").strip()
    if not choice:
        raise ValueError("No database selected.")
    if choice.isdigit():
        index = int(choice) - 1
        if index < 0 or index >= len(db_files):
            raise ValueError("Database selection is out of range.")
        return db_files[index]
    return Path(choice).expanduser()


def get_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})")}


def build_where_clause(columns: set[str], include_reaction_only: bool, include_unclustered: bool) -> tuple[str, list[object]]:
    clauses = [
        "consolidated_tag IS NOT NULL",
        "TRIM(consolidated_tag) != ''",
    ]
    params: list[object] = []

    if not include_unclustered:
        clauses.append("LOWER(TRIM(consolidated_tag)) != ?")
        params.append("unclustered")

    if not include_reaction_only:
        if "is_substantive" in columns:
            clauses.append("is_substantive = 1")
        else:
            clauses.append("LOWER(TRIM(raw_tag)) != ?")
            params.append("reaction only")

    return " AND ".join(clauses), params


def load_pareto_rows(
    db_path: Path,
    top_n: Optional[int],
    include_reaction_only: bool,
    include_unclustered: bool,
) -> list[dict[str, object]]:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "annotation" not in tables:
            raise ValueError(f"{db_path} does not contain an annotation table.")

        columns = get_columns(conn, "annotation")
        where_clause, params = build_where_clause(columns, include_reaction_only, include_unclustered)

        raw_rows = conn.execute(
            f"""
            SELECT
                consolidated_tag AS theme,
                COUNT(*) AS items_in_theme,
                SUM(CASE WHEN item_type = 'post' THEN 1 ELSE 0 END) AS post_items,
                SUM(CASE WHEN item_type = 'comment' THEN 1 ELSE 0 END) AS comment_items
            FROM annotation
            WHERE {where_clause}
            GROUP BY consolidated_tag
            ORDER BY items_in_theme DESC, theme COLLATE NOCASE ASC
            """,
            params,
        ).fetchall()
    finally:
        conn.close()

    if not raw_rows:
        return []

    total = sum(int(row["items_in_theme"]) for row in raw_rows)
    visible_rows = raw_rows[:top_n] if top_n else raw_rows

    cumulative = 0
    rows: list[dict[str, object]] = []
    for rank, row in enumerate(visible_rows, 1):
        count = int(row["items_in_theme"])
        cumulative += count
        rows.append({
            "rank": rank,
            "theme": clean_label(row["theme"]),
            "items_in_theme": count,
            "share_percent": count / total * 100 if total else 0.0,
            "cumulative_items": cumulative,
            "cumulative_share_percent": cumulative / total * 100 if total else 0.0,
            "post_items": int(row["post_items"] or 0),
            "comment_items": int(row["comment_items"] or 0),
            "total_clustered_items": total,
        })
    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def excel_column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def xlsx_cell(value: object, row_idx: int, col_idx: int) -> str:
    ref = f"{excel_column_name(col_idx)}{row_idx}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def worksheet_xml(rows: list[list[object]]) -> str:
    sheet_rows = []
    for row_idx, row in enumerate(rows, 1):
        cells = "".join(xlsx_cell(value, row_idx, col_idx) for col_idx, value in enumerate(row, 1))
        sheet_rows.append(f'<row r="{row_idx}">{cells}</row>')
    cols = "".join(
        f'<col min="{idx}" max="{idx}" width="{width}" customWidth="1"/>'
        for idx, width in enumerate([8, 36, 15, 15, 18, 24, 12, 14, 20], 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<cols>{cols}</cols>"
        "<sheetViews><sheetView workbookViewId=\"0\"><pane ySplit=\"1\" topLeftCell=\"A2\" "
        "activePane=\"bottomLeft\" state=\"frozen\"/></sheetView></sheetViews>"
        f"<sheetData>{''.join(sheet_rows)}</sheetData>"
        '<autoFilter ref="A1:I1"/>'
        "</worksheet>"
    )


def write_xlsx(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_rows: list[list[object]] = [HEADERS]
    for row in rows:
        data_rows.append([row[header] for header in HEADERS])

    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    files = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>'
            "</Relationships>"
        ),
        "docProps/core.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" '
            'xmlns:dcterms="http://purl.org/dc/terms/" '
            'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            "<dc:title>Theme Dominance Pareto Data</dc:title>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>'
            "</cp:coreProperties>"
        ),
        "docProps/app.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" '
            'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
            "<Application>Audience Reception Analysis Pipeline</Application>"
            "</Properties>"
        ),
        "xl/workbook.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Pareto Data" sheetId="1" r:id="rId1"/></sheets>'
            "</workbook>"
        ),
        "xl/_rels/workbook.xml.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            "</Relationships>"
        ),
        "xl/worksheets/sheet1.xml": worksheet_xml(data_rows),
    }

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def output_path_for(db_path: Path, fmt: str, output_arg: Optional[str]) -> Path:
    if output_arg:
        path = Path(output_arg).expanduser()
        return path.with_suffix(f".{fmt}") if path.suffix.lower() != f".{fmt}" else path
    return Path("exports") / f"{db_path.stem}_theme_dominance_pareto_data.{fmt}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export chart-ready theme dominance Pareto data from a workspace SQLite database."
    )
    parser.add_argument("--db", help="Path to workspace .db file. If omitted, choose interactively.")
    parser.add_argument("--format", choices=["csv", "xlsx"], default="csv", help="Output format.")
    parser.add_argument("--output", help="Output path. Defaults to exports/<workspace>_theme_dominance_pareto_data.<format>.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of themes to export. Use 0 for all themes.")
    parser.add_argument("--include-reaction-only", action="store_true", help="Include non-substantive Reaction Only rows.")
    parser.add_argument("--include-unclustered", action="store_true", help="Include unclustered annotations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db).expanduser() if args.db else choose_database(Path.cwd())
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path

    top_n = None if args.top_n == 0 else args.top_n
    rows = load_pareto_rows(
        db_path=db_path,
        top_n=top_n,
        include_reaction_only=args.include_reaction_only,
        include_unclustered=args.include_unclustered,
    )
    if not rows:
        print("No clustered theme data found with the selected filters.")
        return

    output_path = output_path_for(db_path, args.format, args.output)
    if args.format == "csv":
        write_csv(rows, output_path)
    else:
        write_xlsx(rows, output_path)

    print(f"Exported {len(rows)} theme rows from {db_path.name} to {output_path}")
    print("Use 'theme' as the category axis, 'items_in_theme' as bars, and 'cumulative_share_percent' as the line.")


if __name__ == "__main__":
    main()
