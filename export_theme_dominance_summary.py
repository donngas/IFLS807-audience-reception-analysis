import argparse
import csv
import sqlite3
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Optional


HEADERS = [
    "consolidated_tag",
    "cluster_dominance_percent",
    "consolidated_tag_explanation",
]


def clean_text(value: Optional[str], default: str = "") -> str:
    if not value:
        return default
    cleaned = "".join(" " if ord(char) < 32 or 127 <= ord(char) <= 159 else char for char in str(value))
    cleaned = " ".join(cleaned.split())
    return cleaned or default


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


def load_summary_rows(
    db_path: Path,
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
        rows = conn.execute(
            f"""
            SELECT consolidated_tag, cluster_explanation
            FROM annotation
            WHERE {where_clause}
            """,
            params,
        ).fetchall()
    finally:
        conn.close()

    total = len(rows)
    if total == 0:
        return []

    counts: Counter[str] = Counter()
    explanations: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        tag = clean_text(row["consolidated_tag"], "Unclustered")
        explanation = clean_text(row["cluster_explanation"])
        counts[tag] += 1
        if explanation:
            explanations[tag][explanation] += 1

    output_rows = []
    for tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0].lower())):
        explanation = ""
        if explanations[tag]:
            explanation = explanations[tag].most_common(1)[0][0]
        output_rows.append({
            "consolidated_tag": tag,
            "cluster_dominance_percent": round(count / total * 100, 1),
            "consolidated_tag_explanation": explanation,
        })
    return output_rows


def output_path_for(db_path: Path, fmt: str, output_arg: Optional[str]) -> Path:
    if output_arg:
        path = Path(output_arg).expanduser()
        return path.with_suffix(f".{fmt}") if path.suffix.lower() != f".{fmt}" else path
    return Path("exports") / f"{db_path.stem}_theme_dominance_summary.{fmt}"


def write_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Consolidated tag | Cluster dominance percentage | Consolidated tag explanation |",
        "|---|---:|---|",
    ]
    for row in rows:
        tag = str(row["consolidated_tag"]).replace("|", "\\|")
        explanation = str(row["consolidated_tag_explanation"]).replace("|", "\\|")
        lines.append(f"| {tag} | {row['cluster_dominance_percent']:.1f}% | {explanation} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def excel_column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def xlsx_cell(value: object, row_idx: int, col_idx: int) -> str:
    ref = f"{excel_column_name(col_idx)}{row_idx}"
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
        for idx, width in enumerate([32, 24, 95], 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<cols>{cols}</cols>"
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
        'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        f"<sheetData>{''.join(sheet_rows)}</sheetData>"
        '<autoFilter ref="A1:C1"/>'
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
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            "<dc:title>Theme Dominance Summary</dc:title>"
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>'
            "</cp:coreProperties>"
        ),
        "docProps/app.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
            "<Application>Audience Reception Analysis Pipeline</Application>"
            "</Properties>"
        ),
        "xl/workbook.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Dominance Summary" sheetId="1" r:id="rId1"/></sheets>'
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export consolidated theme dominance percentages and cluster explanations from a workspace SQLite database."
    )
    parser.add_argument("--db", help="Path to workspace .db file. If omitted, choose interactively.")
    parser.add_argument("--format", choices=["csv", "xlsx", "md"], default="csv", help="Output format.")
    parser.add_argument("--output", help="Output path. Defaults to exports/<workspace>_theme_dominance_summary.<format>.")
    parser.add_argument("--include-reaction-only", action="store_true", help="Include non-substantive Reaction Only rows.")
    parser.add_argument("--include-unclustered", action="store_true", help="Include unclustered annotations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db).expanduser() if args.db else choose_database(Path.cwd())
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path

    rows = load_summary_rows(
        db_path=db_path,
        include_reaction_only=args.include_reaction_only,
        include_unclustered=args.include_unclustered,
    )
    if not rows:
        print("No clustered theme data found with the selected filters.")
        return

    output_path = output_path_for(db_path, args.format, args.output)
    if args.format == "csv":
        write_csv(rows, output_path)
    elif args.format == "xlsx":
        write_xlsx(rows, output_path)
    else:
        write_markdown(rows, output_path)

    print(f"Exported {len(rows)} theme summary rows from {db_path.name} to {output_path}")


if __name__ == "__main__":
    main()
