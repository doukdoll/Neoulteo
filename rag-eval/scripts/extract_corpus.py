"""MySQL 덤프(SSAFY_TRIP_Dump.sql)에서 관광지 코퍼스를 JSONL로 추출한다.

Spring 쪽 RagSearchService와 동일한 문서 구성(관광지명/주소/분류/설명)을 재현하기 위해
attractions + contenttypes + sidos + guguns 를 조인한 결과를 만든다.

usage: python scripts/extract_corpus.py ../docs/database/SSAFY_TRIP_Dump.sql data/corpus.jsonl
"""
import json
import re
import sys
from pathlib import Path

ATTR_COLS = [
    "id", "content_id", "title", "content_type_id", "area_code", "si_gun_gu_code",
    "first_image1", "first_image2", "mlevel", "latitude", "longitude", "tel",
    "addr1", "addr2", "homepage", "overview",
]


def parse_values(s: str, i: int):
    """INSERT ... VALUES 뒤의 (..),(..); 튜플들을 파싱한다. MySQL 문자열 이스케이프 처리."""
    rows, row, n = [], [], len(s)
    while i < n:
        c = s[i]
        if c == "(":
            row, i = [], i + 1
            while True:
                c = s[i]
                if c == "'":
                    i += 1
                    buf = []
                    while True:
                        c = s[i]
                        if c == "\\":
                            nxt = s[i + 1]
                            buf.append({"n": "\n", "r": "\r", "t": "\t", "0": "\0"}.get(nxt, nxt))
                            i += 2
                        elif c == "'":
                            if s[i + 1] == "'":
                                buf.append("'")
                                i += 2
                            else:
                                i += 1
                                break
                        else:
                            buf.append(c)
                            i += 1
                    row.append("".join(buf))
                else:
                    j = i
                    while s[j] not in ",)":
                        j += 1
                    tok = s[i:j].strip()
                    row.append(None if tok == "NULL" else tok)
                    i = j
                if s[i] == ",":
                    i += 1
                    continue
                if s[i] == ")":
                    i += 1
                    rows.append(row)
                    break
        elif c == ";":
            return rows, i + 1
        else:
            i += 1
    return rows, i


def table_rows(sql: str, table: str):
    out = []
    for m in re.finditer(rf"INSERT INTO `{table}` VALUES ", sql):
        rows, _ = parse_values(sql, m.end())
        out.extend(rows)
    return out


def clean(v):
    if v is None:
        return ""
    v = re.sub(r"<br\s*/?>", " ", v, flags=re.I)
    v = re.sub(r"<[^>]+>", " ", v)
    return re.sub(r"\s+", " ", v).strip()


def main(dump_path: str, out_path: str):
    sql = Path(dump_path).read_text(encoding="utf-8", errors="ignore")
    ctypes = {int(r[0]): r[1] for r in table_rows(sql, "contenttypes")}
    sidos = {int(r[1]): r[2] for r in table_rows(sql, "sidos")}
    guguns = {(int(r[1]), int(r[2])): r[3] for r in table_rows(sql, "guguns")}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in table_rows(sql, "attractions"):
            d = dict(zip(ATTR_COLS, r))
            title = clean(d["title"])
            if not title:
                continue
            area = int(d["area_code"]) if d["area_code"] else None
            sigungu = int(d["si_gun_gu_code"]) if d["si_gun_gu_code"] else None
            ctype = int(d["content_type_id"]) if d["content_type_id"] else None
            doc = {
                "content_id": int(d["content_id"]),
                "title": title,
                "address": clean(d["addr1"]),
                "tel": clean(d["tel"]),
                "content_type_id": ctype,
                "content_type_name": ctypes.get(ctype, ""),
                "area_code": area,
                "sigungu_code": sigungu,
                "sido": sidos.get(area, ""),
                "gugun": guguns.get((area, sigungu), ""),
                "overview": clean(d["overview"]),
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} docs -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
