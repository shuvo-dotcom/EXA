#!/usr/bin/env python3
"""plexos_table_extractor.py  – NaN‑safe, no‑argparse, **all master tables**

Adds optional **t_property_group_id filtering** for *t_property*.
Run with:
    python plexos_table_extractor.py master.xml t_property property_group=3 csv=1
or via the interactive prompt when you choose table (8).
"""

from __future__ import annotations

import sys
import pathlib
from typing import Dict, List, Set

import pandas as pd
from lxml import etree

DEFAULT_XML = r"C:\Program Files\Energy Exemplar\PLEXOS 10.0 API\master.xml"
VALID_TABLES = {
                    "t_config",
                    "t_class_group",
                    "t_class",
                    "t_unit",
                    "t_attribute",
                    "t_collection",
                    "t_property_group",
                    "t_property",
                    "t_collection_report",
                    "t_property_report",
                }

# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────

def _smart_cast(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dropna().str.fullmatch(r"-?\d+").all():
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    if s.dropna().isin({"true", "false"}).all():
        return s.map({"true": True, "false": False}).astype("boolean")
    return s

def extract_table(xml_path: str | pathlib.Path, table_tag: str, *,  filters: Dict[str, Set[str]] | None = None) -> pd.DataFrame:
    records: List[Dict[str, str]] = []
    xml_path = pathlib.Path(xml_path)
    filters = filters or {}
    qname = f"{{*}}{table_tag}"
    for _evt, elem in etree.iterparse(str(xml_path), events=("end",), tag=(table_tag, qname)):
        row = {c.tag.split('}',1)[-1]: (c.text or "") for c in elem}
        ok = True
        for col, allowed in filters.items():
            if col == "class_ids":
                ok &= row.get("parent_class_id") in allowed and row.get("child_class_id") in allowed
            else:
                ok &= row.get(col) in allowed
            if not ok: break
        if ok:
            records.append(row)
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    return pd.DataFrame(records).apply(_smart_cast)

# ──────────────────────────────────────────────────────────────────────────
# high‑level export
# ──────────────────────────────────────────────────────────────────────────

def export_table(xml_path: str | pathlib.Path,
                 table: str,
                 *,
                 class_group: int | None = None,
                 class_ids: Set[int] | None = None,
                 property_group: int | None = None,
                 unit_id: int | None = None,
                 collection_id: int | None = None,
                 csv: bool | str = False,
                 preview: bool = False) -> pd.DataFrame:
    filters: Dict[str, Set[str]] = {}

    if class_group is not None and table == "t_class":
        filters["class_group_id"] = {str(class_group)}

#    if table == "t_collection":
#        filters["class_ids"] = {str(i) for i in class_ids}

    if property_group is not None and table == "t_property":
        filters["property_group_id"] = {str(property_group)}
        filters["collection_id"] = {str(collection_id)}

    if table == "t_unit":
        filters["unit_id"] = {str(unit_id)}

    df = extract_table(xml_path, table, filters=filters)
    # print(df.head())
    return df

# ──────────────────────────────────────────────────────────────────────────
# interactive mode
# ──────────────────────────────────────────────────────────────────────────

def interactive_mode(table_choice: str, grp = None, ids = None, pg = None, class_id_1 = None, class_id_2 = None, unit_id = None, collection_id = None) -> None:
    # print("\n🟢  PLEXOS Table Extractor – Interactive Mode\n"+"═"*50)
    xml = DEFAULT_XML
    while not pathlib.Path(xml).exists():
        print("⚠ file not found – try again\n")
        # In this version, we assume DEFAULT_XML exists or the user will fix it externally.

    table_list = sorted(VALID_TABLES)

    if table_choice not in table_list:
        print(f"⚠ '{table_choice}' is not a valid table name.")
        print("Valid tables are:", ", ".join(table_list))
        return

    table = table_choice

    kwargs: Dict[str,object] = {}
    if table=="t_class":
        if grp: 
            kwargs["class_group"] = int(grp)
        
    if table=="t_collection":
        kwargs["class_ids"]={class_id_1, class_id_2}

    if table=="t_property":
        if pg: 
            kwargs["property_group"] = int(pg)
            kwargs["collection_id"] = int(collection_id)

    if table == "t_unit":
        if unit_id:
            kwargs["unit_id"] = int(unit_id)

    return export_table(xml, table, **kwargs)

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    class_id = 2
    collection_id = 416 #generator fuels
    x = interactive_mode('t_class')
    # x = x[x['collection_id'] == 2]
    
    # x = x[x['parent_class_id'] == class_id]
    # x = x[x['min_count'] == 1]
    # property_data = x[(x['parent_class_id'] == class_id) | (x['child_class_id'] == class_id)]
    # print(x['name'].unique())
    # print(x[['name', 'class_group_id', 'description']].to_dict('records'))
    collection_data = x[(x['parent_class_id'] == class_id) | (x['child_class_id'] == class_id)]
    # map collection_id -> {'name': ..., 'parent_class_id': ...}
    collection_names = collection_data.set_index('collection_id')[['name', 'parent_class_id', 'child_class_id','description']].to_dict(orient='index')

    property_data = x[x['parent_class_id'] == class_id]
    property_data_2 = x[x['child_class_id'] == class_id]

    print(x)