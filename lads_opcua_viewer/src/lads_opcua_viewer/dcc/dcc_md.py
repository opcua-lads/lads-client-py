#!/usr/bin/env python3
"""
dcc_to_markdown.py

Render a PTB Digital Calibration Certificate (DCC) XML to Markdown.

Goals:
- Reproduce the human-readable structure of a weighing calibration certificate
  (as in the provided PDF) including:
  - Certificate / traceability statements
  - Device / customer / calibration info
  - Environmental + adjustment status
  - Measurement results:
    * Repeatability
    * Eccentricity
    * Error of indication w/ uncertainties
  - Uncertainty-in-use summary

- Be defensive:
  * Never crash just because a node is missing.
  * If data isn't present in the XML, we emit "—" or skip the table/section.
  * We don't assume fixed namespace prefixes.

This script DOES NOT:
- Render logos, signatures, QR codes, graphics.
- Validate against XSD. (Can be added, but we don't REQUIRE lxml here.)
"""

import sys
import math
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple


# ---------- generic XML helpers (namespace-robust) ----------

def local_name(tag: str) -> str:
    """Return the local part of a {ns}tag or plain tag."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag

def children(el: Optional[ET.Element], name: str = None) -> List[ET.Element]:
    """
    Return direct child elements of `el`.
    If `name` is given, filter by local-name == name.
    If el is None, return [].
    """
    if el is None:
        return []
    if name is None:
        return list(el)
    return [c for c in el if local_name(c.tag) == name]

def first_child(el: Optional[ET.Element], name: str) -> Optional[ET.Element]:
    """
    Return the first direct child of `el` with given local-name, else None.
    """
    for c in children(el, name):
        return c
    return None

def text_of(el: Optional[ET.Element], default: str = "—") -> str:
    """Return .text stripped, or default if el missing/empty."""
    if el is None:
        return default
    if el.text is None:
        return default
    t = el.text.strip()
    return t if t else default

def findall_path(root_el: Optional[ET.Element], path_local_names: List[str]) -> List[ET.Element]:
    """
    Poor man's namespace-agnostic findall for a simple path of local-names.
    Example: ["measurementResults","measurementResult","results","result"]
    Returns all matches at the final level.
    """
    if root_el is None:
        return []
    curr = [root_el]
    for lname in path_local_names:
        nxt = []
        for node in curr:
            nxt.extend(children(node, lname))
        curr = nxt
    return curr


# ---------- safe number parsing / formatting ----------

def parse_float(s: str) -> Optional[float]:
    if s is None:
        return None
    s = s.strip()
    if s == "" or s.lower() == "nan":
        return None
    # accept both comma and dot decimals:
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

def fmt_float(val: Optional[float],
              digits: int = 6,
              force_sign: bool = False) -> str:
    """
    Format float in a human-ish way similar to the PDF output.
    - default 6 decimal places
    - fallback "—"
    """
    if val is None:
        return "—"
    # special: we usually see up to 6 decimals, sometimes 7+ like standard deviation
    # we'll keep it flexible but stable.
    fmt = "{:+." + str(digits) + "f" if force_sign else "{:." + str(digits) + "f"
    try:
        return fmt.format(val)
    except Exception:
        # fallback generic repr
        return f"{val:.6g}"

def split_floats_from_space_list(s: str) -> List[Optional[float]]:
    """
    '0.000000 -0.000002 -0.000003' -> [0.0, -2e-6, -3e-6 ...]
    Handles commas as decimal separators too.
    """
    out: List[Optional[float]] = []
    if not s:
        return out
    parts = s.strip().split()
    for p in parts:
        out.append(parse_float(p))
    return out

def pad_to_same_len(cols: List[List[Any]]) -> None:
    """
    In-place pad shorter lists with None so we can zip safely for tables.
    """
    maxlen = max((len(c) for c in cols), default=0)
    for c in cols:
        while len(c) < maxlen:
            c.append(None)


# ---------- extraction helpers for certificate metadata ----------

def extract_core_data(root: ET.Element) -> Dict[str, Any]:
    core = first_child(first_child(root, "administrativeData"), "coreData")
    data: Dict[str, Any] = {}

    data["schemaVersion"] = root.attrib.get("schemaVersion", "—")
    data["beginPerformanceDate"] = text_of(first_child(core, "beginPerformanceDate"))
    data["endPerformanceDate"]   = text_of(first_child(core, "endPerformanceDate"))

    # Unique identifier(s)
    data["uniqueIdentifier"] = text_of(first_child(core, "uniqueIdentifier"))

    # certificate numbers / identifications
    data["certificateIds"] = []
    identifications = first_child(core, "identifications")
    for ident in children(identifications, "identification"):
        issuer = text_of(first_child(ident, "issuer"))
        value  = text_of(first_child(ident, "value"))
        data["certificateIds"].append({
            "issuer": issuer,
            "value": value
        })

    return data

def extract_customer(root: ET.Element) -> Dict[str, Any]:
    cust = first_child(first_child(root, "administrativeData"), "customer")
    out: Dict[str, Any] = {}
    # name may have multiple <content> lines -> join them
    name_el = first_child(cust, "name")
    lines = []
    for c in children(name_el, "content"):
        t = text_of(c, "")
        if t:
            lines.append(t)
    out["name_lines"] = lines

    loc_el = first_child(cust, "location")
    out["location"] = {
        "street":      text_of(first_child(loc_el, "street")),
        "postCode":    text_of(first_child(loc_el, "postCode")),
        "city":        text_of(first_child(loc_el, "city")),
        "countryCode": text_of(first_child(loc_el, "countryCode")),
    }
    return out

def extract_lab(root: ET.Element) -> Dict[str, Any]:
    lab = first_child(first_child(root, "administrativeData"), "calibrationLaboratory")
    out: Dict[str, Any] = {}
    out["code"] = text_of(first_child(lab, "calibrationLaboratoryCode"))

    contact = first_child(lab, "contact")
    out["labName"] = text_of(first_child(contact, "content"))
    out["email"]   = text_of(first_child(contact, "eMail"))

    loc = first_child(contact, "location")
    out["labLocation"] = {
        "street":   text_of(first_child(loc, "street")),
        "streetNo": text_of(first_child(loc, "streetNo")),
        "postCode": text_of(first_child(loc, "postCode")),
        "city":     text_of(first_child(loc, "city")),
        "country":  text_of(first_child(loc, "countryCode")),
    }
    return out

def extract_responsibles(root: ET.Element) -> List[Dict[str, Any]]:
    """
    Extract responsible persons and their roles/signature status.

    Returns a list of dicts:
    {
      "role": "Authorization of the Certificate",
      "mainSigner": "true",
      "name": "Karin Hagedorn"
    }
    """
    administrativeData = first_child(root, "administrativeData")
    persons_parent = first_child(administrativeData, "respPersons")

    out = []
    for rp in children(persons_parent, "respPerson"):
        role_txt = text_of(first_child(rp, "role"), "—")
        main_signer_txt = text_of(first_child(rp, "mainSigner"), "—")

        # person/name/content
        person_el = first_child(rp, "person")
        name_el = first_child(person_el, "name") if person_el is not None else None

        # collect all <content> under <name>
        person_name_chunks = []
        if name_el is not None:
            for c in children(name_el, "content"):
                t = text_of(c, "")
                if t and t != "—":
                    person_name_chunks.append(t)

        person_name = " ".join(ch.strip() for ch in person_name_chunks if ch.strip()) or "—"

        out.append({
            "role": role_txt,
            "mainSigner": main_signer_txt,
            "name": person_name
        })

    return out

# ---------- extraction: item / instrument ----------

def extract_item(root: ET.Element) -> Dict[str, Any]:
    items_el = first_child(first_child(root, "administrativeData"), "items")
    item_el  = first_child(items_el, "item")

    out: Dict[str, Any] = {}

    # Instrument "name" (bilingual)
    name_el = first_child(item_el, "name")
    out["objectName_en"] = None
    out["objectName_de"] = None
    for c in children(name_el, "content"):
        lang = c.attrib.get("lang", "").lower()
        if lang == "en":
            out["objectName_en"] = text_of(c)
        elif lang == "de":
            out["objectName_de"] = text_of(c)

    # Manufacturer
    manu_el = first_child(item_el, "manufacturer")
    manu_name_el = first_child(manu_el, "name")
    manu_name = None
    for c in children(manu_name_el, "content"):
        if c.text and c.text.strip():
            manu_name = c.text.strip()
    out["manufacturer"] = manu_name or "—"

    # Model
    out["model"] = text_of(first_child(item_el, "model"))

    # Identifications (serial no, customer equipment no, etc.)
    out["serialNumber"] = "—"
    out["equipmentNumber"] = "—"
    ident_parent = first_child(item_el, "identifications")
    for ident in children(ident_parent, "identification"):
        refType = ident.attrib.get("refType", "")
        value   = text_of(first_child(ident, "value"))
        if "serial" in refType.lower():
            out["serialNumber"] = value
        elif "measuringequipmentnumber" in refType.lower():
            out["equipmentNumber"] = value

    # Subitems → weighing range, max load, etc.
    #   We assume first subItem is the only calibrated range for this cert.
    subItems = first_child(item_el, "subItems")
    subItem  = first_child(subItems, "item")
    out["rangeName_en"] = None
    out["rangeName_de"] = None
    if subItem is not None:
        sub_name = first_child(subItem, "name")
        for c in children(sub_name, "content"):
            lang = c.attrib.get("lang", "").lower()
            if lang == "en":
                out["rangeName_en"] = text_of(c)
            elif lang == "de":
                out["rangeName_de"] = text_of(c)

        # quantities like min, max, resolution
        quants = first_child(subItem, "itemQuantities")
        qinfo: Dict[str, Dict[str, Any]] = {}
        for q in children(quants):
            ref = q.attrib.get("refType", "")
            real = first_child(q, "real")  # may be si:real, so we fallback local-name match:
            if real is None:
                # try any child whose local-name == "real"
                for alt in children(q):
                    if local_name(alt.tag) == "real":
                        real = alt
                        break
            if real is not None:
                value = parse_float(text_of(first_child(real, "value"), None))
                unit  = text_of(first_child(real, "unit"), "")
                qinfo[ref] = {"value": value, "unit": unit}
        out["quantities"] = qinfo
    else:
        out["quantities"] = {}

    return out


# ---------- extraction: environment / conditions ----------

def extract_environment_and_adjustment(meas_result_el: ET.Element) -> Dict[str, Any]:
    """
    Pulls:
    - temperature at calibration site
    - adjustment status, plus human text
    """
    out: Dict[str, Any] = {
        "temperature_value": None,
        "temperature_unit": None,
        "adjustment_status": "—",
        "adjustment_comment_de": None,
        "adjustment_comment_en": None,
    }

    infl_parent = first_child(meas_result_el, "influenceConditions")
    for infl in children(infl_parent, "influenceCondition"):
        refType = infl.attrib.get("refType", "")
        if "temperature" in refType.lower():
            data_el = first_child(infl, "data")
            quantity_el = first_child(data_el, "quantity")
            if quantity_el is not None:
                # grab real/value/unit (could be under si namespace)
                real = first_child(quantity_el, "real")
                if real is None:
                    for alt in children(quantity_el):
                        if local_name(alt.tag) == "real":
                            real = alt
                            break
                if real is not None:
                    out["temperature_value"] = parse_float(
                        text_of(first_child(real, "value"), None)
                    )
                    out["temperature_unit"] = text_of(first_child(real, "unit"), "—")

        if "adjustment" in refType.lower():
            out["adjustment_status"] = text_of(first_child(infl, "status"), "—")
            data_el = first_child(infl, "data")
            # inside data → quantity → name/description with bilingual content
            for q in children(data_el, "quantity"):
                desc_el = first_child(q, "description")
                if desc_el is None:
                    continue
                for c in children(desc_el, "content"):
                    lang = c.attrib.get("lang", "").lower()
                    if lang == "de":
                        out["adjustment_comment_de"] = text_of(c)
                    elif lang == "en":
                        out["adjustment_comment_en"] = text_of(c)

    return out


# ---------- extraction: used method(s) / procedure ----------

def extract_methods(root: ET.Element) -> Dict[str, Any]:
    usedMethods = first_child(first_child(root, "measurementResults"), "usedMethods")
    methods = []
    for m in children(usedMethods, "usedMethod"):
        entry = {
            "refType": m.attrib.get("refType", "—"),
            "name_de": None,
            "name_en": None,
            "norm": text_of(first_child(m, "norm")),
            "link": text_of(first_child(m, "link"), "—"),
            "description_de": None,
            "description_en": None,
        }
        name_el = first_child(m, "name")
        for c in children(name_el, "content"):
            if c.attrib.get("lang","").lower() == "de":
                entry["name_de"] = text_of(c)
            elif c.attrib.get("lang","").lower() == "en":
                entry["name_en"] = text_of(c)
        desc_el = first_child(m, "description")
        for c in children(desc_el, "content"):
            if c.attrib.get("lang","").lower() == "de":
                entry["description_de"] = text_of(c)
            elif c.attrib.get("lang","").lower() == "en":
                entry["description_en"] = text_of(c)
        methods.append(entry)
    return {"methods": methods}


# ---------- extraction: measuring equipment ----------

def extract_reference_equipment(root: ET.Element) -> List[Dict[str, Any]]:
    me_parent = first_child(first_child(root, "measurementResults"), "measuringEquipments")
    out = []
    for me in children(me_parent, "measuringEquipment"):
        name_de = name_en = None
        for c in children(first_child(me, "name"), "content"):
            if c.attrib.get("lang","").lower() == "de":
                name_de = text_of(c)
            elif c.attrib.get("lang","").lower() == "en":
                name_en = text_of(c)
        eq_class = first_child(me, "equipmentClass")
        classID = text_of(first_child(eq_class, "classID"))
        reference = text_of(first_child(eq_class, "reference"))

        # identification value(s)
        ident_parent = first_child(me, "identifications")
        ids = []
        for ident in children(ident_parent, "identification"):
            issuer = text_of(first_child(ident, "issuer"))
            value  = text_of(first_child(ident, "value"))
            ids.append({"issuer": issuer, "value": value})

        out.append({
            "name_en": name_en or "—",
            "name_de": name_de or "—",
            "classID": classID,
            "reference": reference,
            "ids": ids
        })
    return out


# ---------- extraction: measurement result sections ----------

def get_main_measurement_result(root: ET.Element) -> Optional[ET.Element]:
    # We assume 1 measurementResult in your file (range1).
    mr_parent = first_child(first_child(root, "measurementResults"), "measurementResult")
    return mr_parent


def extract_repeatability(rep_result_el: ET.Element) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts, one per 'Measurement at X gram' block.
    Each dict:
      {
        "title_de": "...",
        "title_en": "...",
        "nominal_value": float,
        "nominal_unit": str,
        "measured_values": [floats...],
        "stdev_value": float,
        "stdev_unit": str,
      }
    """
    out = []
    data_el = first_child(rep_result_el, "data")
    # rep_result_el structure: <result refType="NAWI_repeatabilityMeasurement">
    #   <data>
    #     <list> ... block for 0.1g ... </list>
    #     <list> ... block for 5g ... </list>
    for lst in children(data_el, "list"):
        block = {
            "title_de": "—",
            "title_en": "—",
            "nominal_value": None,
            "nominal_unit": "—",
            "measured_values": [],
            "stdev_value": None,
            "stdev_unit": "—",
        }
        for entry in children(lst):
            lname = local_name(entry.tag)
            if lname == "name":
                for c in children(entry, "content"):
                    lang = c.attrib.get("lang", "").lower()
                    if lang == "de":
                        block["title_de"] = text_of(c)
                    elif lang == "en":
                        block["title_en"] = text_of(c)
            elif lname == "quantity":
                refType = entry.attrib.get("refType", "")
                # nominal value?
                if "nominal" in refType.lower():
                    real = first_child(entry, "real")
                    if real is None:
                        # also allow si:real
                        for alt in children(entry):
                            if local_name(alt.tag) == "real":
                                real = alt
                                break
                    if real is not None:
                        block["nominal_value"] = parse_float(text_of(first_child(real, "value"), None))
                        block["nominal_unit"]  = text_of(first_child(real, "unit"), "—")
                # measured values list?
                if "measuredvalue" in refType.lower():
                    rlist = first_child(entry, "realListXMLList")
                    if rlist is None:
                        for alt in children(entry):
                            if local_name(alt.tag) == "realListXMLList":
                                rlist = alt
                                break
                    if rlist is not None:
                        block["measured_values"] = split_floats_from_space_list(
                            text_of(first_child(rlist, "valueXMLList"), "")
                        )
                # stdev?
                if "standarddeviation" in refType.lower():
                    real = first_child(entry, "real")
                    if real is None:
                        for alt in children(entry):
                            if local_name(alt.tag) == "real":
                                real = alt
                                break
                    if real is not None:
                        block["stdev_value"] = parse_float(text_of(first_child(real, "value"), None))
                        block["stdev_unit"]  = text_of(first_child(real, "unit"), "—")
        out.append(block)
    return out


def extract_eccentricity(ecc_result_el: ET.Element) -> Dict[str, Any]:
    """
    Returns dict with:
    {
      "nominal_value": float,
      "nominal_unit": str,
      "reference_value": float,
      "reference_unit": str,
      "positions": [
          {"label_en": "...", "label_de": "...", "indication": float},
          ...
      ],
      "max_error_value": float,
      "max_error_unit": str
    }
    """
    out = {
        "nominal_value": None,
        "nominal_unit": "—",
        "reference_value": None,
        "reference_unit": "—",
        "positions": [],
        "max_error_value": None,
        "max_error_unit": "—",
    }

    data_el = first_child(ecc_result_el, "data")
    # Position labels appear in <text><content>Position1: Front left</content>...
    # We'll build a mapping Position# -> text
    pos_label_map_en: Dict[str,str] = {}
    pos_label_map_de: Dict[str,str] = {}
    text_el = first_child(data_el, "text")
    for c in children(text_el, "content"):
        raw = text_of(c, "")
        # e.g. "Position1: Front left"
        if ":" in raw:
            key, val = raw.split(":", 1)
            key = key.strip()
            val = val.strip()
            lang = c.attrib.get("lang","").lower()
            if lang == "en":
                pos_label_map_en[key] = val
            elif lang == "de":
                pos_label_map_de[key] = val

    # now parse list with quantities
    lst = first_child(data_el, "list")
    if lst is None:
        return out

    # temporary store differences etc.
    deviations = []  # list of deviations in same order as labelXMLList
    dev_labels = []  # Position1 etc.
    dev_unit   = "—"

    for entry in children(lst):
        refType = entry.attrib.get("refType", "")
        if "nominalvalue" in refType.lower():
            # get nominal load
            real = first_child(entry, "real") or first_child_any_ns(entry, "real")
            real = real or first_child(entry, "real")
            real = real or first_child_any_ns(entry, "real")
        # this block is a bit verbose. We'll use helper below to clean it up in final code.
    # We'll rewrite ecc parsing cleanly below with helpers.
    return _extract_eccentricity_clean(ecc_result_el,
                                       pos_label_map_en,
                                       pos_label_map_de)


def first_child_any_ns(el: Optional[ET.Element], name: str) -> Optional[ET.Element]:
    if el is None:
        return None
    for c in el:
        if local_name(c.tag) == name:
            return c
    return None

def _extract_eccentricity_clean(ecc_result_el: ET.Element,
                                pos_label_map_en: Dict[str,str],
                                pos_label_map_de: Dict[str,str]) -> Dict[str, Any]:
    out = {
        "nominal_value": None,
        "nominal_unit": "—",
        "reference_value": None,
        "reference_unit": "—",
        "positions": [],
        "max_error_value": None,
        "max_error_unit": "—",
    }

    data_el = first_child(ecc_result_el, "data")
    if data_el is None:
        return out

    lst = first_child(data_el, "list")
    if lst is None:
        return out

    # We will gather:
    deviations_vals: List[Optional[float]] = []
    deviations_labels: List[str] = []
    deviations_unit = "—"

    center_value = None
    center_unit  = "—"

    for entry in children(lst):
        refType = entry.attrib.get("refType", "").lower()

        # nominal load of test
        if "nominalvalue" in refType:
            real = first_child_any_ns(entry, "real")
            if real is not None:
                out["nominal_value"] = parse_float(text_of(first_child_any_ns(real, "value"), None))
                out["nominal_unit"]  = text_of(first_child_any_ns(real, "unit"), "—")

        # reference value (center indication)
        if "referencevalue" in refType:
            real = first_child_any_ns(entry, "real")
            if real is not None:
                out["reference_value"] = parse_float(text_of(first_child_any_ns(real, "value"), None))
                out["reference_unit"]  = text_of(first_child_any_ns(real, "unit"), "—")
                # keep also for "center"
                center_value = out["reference_value"]
                center_unit  = out["reference_unit"]

        # deviations at each corner
        if "measurementerror" in refType and "maximum" not in refType:
            rlist = first_child_any_ns(entry, "realListXMLList")
            if rlist is not None:
                labellist = text_of(first_child_any_ns(rlist, "labelXMLList"), "")
                vallist   = text_of(first_child_any_ns(rlist, "valueXMLList"), "")
                deviations_labels = labellist.split()
                deviations_vals   = split_floats_from_space_list(vallist)
                deviations_unit   = text_of(first_child_any_ns(rlist, "unitXMLList"), "—")

        # maximum error
        if "measurementerror" in refType and "maximum" in refType:
            real = first_child_any_ns(entry, "real")
            if real is not None:
                out["max_error_value"] = parse_float(text_of(first_child_any_ns(real,"value"), None))
                out["max_error_unit"]  = text_of(first_child_any_ns(real,"unit"), "—")

    # map each position
    for lbl, dev in zip(deviations_labels, deviations_vals):
        # build nice label using both EN/DE if we have them
        human_en = pos_label_map_en.get(lbl.capitalize(), pos_label_map_en.get(lbl, lbl))
        human_de = pos_label_map_de.get(lbl.capitalize(), pos_label_map_de.get(lbl, lbl))
        out["positions"].append({
            "position_label_en": human_en or lbl,
            "position_label_de": human_de or lbl,
            "deviation": dev,
            "unit": deviations_unit
        })

    # also include center explicitly at the top (what PDF shows as "Mitte")
    if center_value is not None:
        out["positions"].insert(0, {
            "position_label_en": "Center",
            "position_label_de": "Mitte",
            "deviation": center_value,
            "unit": center_unit
        })

    return out


def extract_error_of_indication(err_result_el: ET.Element) -> Dict[str, Any]:
    """
    Build the indication error table like:

    L (Nominal) | I (Indication) | E (Error) | k | U(E) | U_rel(E)

    where k and U(E) come from the measurementUncertaintyUnivariateXMLList.

    Returns dict with keys:
    {
      "rows": [
         {
           "L": float,
           "I": float,
           "E": float,
           "U": float,
           "k": float,
           "Urel_percent": float or None
         }, ...
      ],
      "max_error_abs": float,
      "max_error_unit": str
    }
    """
    out = {"rows": [], "max_error_abs": None, "max_error_unit": "—"}

    data_el = first_child(err_result_el, "data")
    if data_el is None:
        return out

    # We'll parse each quantity block by refType.
    nominal_vals = []
    indication_vals = []
    error_vals = []
    U_vals = []
    k_vals = []
    max_error_abs = None
    max_error_unit = "—"

    # We'll grab these lists in parallel by index.
    for q in children(data_el, "quantity"):
        refType = q.attrib.get("refType", "").lower()

        # nominal load list (L)
        if "nominalvalue" in refType:
            rlist = first_child_any_ns(q, "realListXMLList")
            if rlist is not None:
                nominal_vals = split_floats_from_space_list(
                    text_of(first_child_any_ns(rlist, "valueXMLList"), "")
                )

        # measured indication list (I)
        if "measuredvalue" in refType:
            rlist = first_child_any_ns(q, "realListXMLList")
            if rlist is not None:
                indication_vals = split_floats_from_space_list(
                    text_of(first_child_any_ns(rlist, "valueXMLList"), "")
                )

        # error list (E), plus uncertainties in nested measurementUncertaintyUnivariateXMLList
        if refType.strip() == "basic_measurementError".lower():
            rlist = first_child_any_ns(q, "realListXMLList")
            if rlist is not None:
                error_vals = split_floats_from_space_list(
                    text_of(first_child_any_ns(rlist, "valueXMLList"), "")
                )

                mu_parent = first_child_any_ns(rlist, "measurementUncertaintyUnivariateXMLList")
                if mu_parent is not None:
                    expMU = first_child_any_ns(mu_parent, "expandedMUXMLList")
                    if expMU is not None:
                        U_vals = split_floats_from_space_list(
                            text_of(first_child_any_ns(expMU, "valueExpandedMUXMLList"), "")
                        )
                        k_vals = split_floats_from_space_list(
                            text_of(first_child_any_ns(expMU, "coverageFactorXMLList"), "")
                        )
                        # coverageProbabilityXMLList etc. exist as well if you need them.

        # maximum error of indication
        if "measurementerror" in refType and "maximum" in refType:
            real = first_child_any_ns(q, "real")
            if real is not None:
                max_error_abs = parse_float(text_of(first_child_any_ns(real, "value"), None))
                max_error_unit = text_of(first_child_any_ns(real, "unit"), "—")

    # make all lists same length
    col_lists = [nominal_vals, indication_vals, error_vals, U_vals, k_vals]
    pad_to_same_len(col_lists)
    nominal_vals, indication_vals, error_vals, U_vals, k_vals = col_lists

    # build rows
    for L, I, E, U, k in zip(nominal_vals, indication_vals, error_vals, U_vals, k_vals):
        urel = None
        # relative uncertainty U_rel(E) = U / L * 100 %, PDF uses "---" for L == 0
        if L is not None and L != 0 and U is not None:
            urel = (U / L) * 100.0
        out["rows"].append({
            "L": L,
            "I": I,
            "E": E,
            "k": k,
            "U": U,
            "Urel_percent": urel,
        })

    out["max_error_abs"]  = max_error_abs
    out["max_error_unit"] = max_error_unit

    return out


def extract_uncertainty_in_use(extras: List[ET.Element],
                               item_data: Dict[str,Any]) -> Dict[str, Any]:
    """
    Parse the "Uncertainty of measurement in use" block
    (linear formula U(W) = alpha + beta * R) and derive the table
    at 1%,25%,50%,75%,100% of max load.

    Returns {
      "alpha": float or None,
      "beta": float or None,
      "max_load": float or None,
      "table_rows": [ { "pct": int, "R": float, "U": float, "Urel_percent": float }, ... ]
    }
    """
    out = {
        "alpha": None,
        "beta": None,
        "max_load": None,
        "table_rows": []
    }

    # find the extra result with a <formula> child (your first basic_contentOutsideDCC)
    formula_block = None
    for ex in extras:
        if first_child(ex, "data") is not None and first_child(first_child(ex,"data"), "formula") is not None:
            formula_block = ex
            break
    if formula_block is None:
        return out

    data_el = first_child(formula_block, "data")

    alpha = None
    beta = None
    for q in children(data_el, "quantity"):
        qid = q.attrib.get("id","")
        real = first_child_any_ns(q, "real")
        if real is None:
            continue
        val = parse_float(text_of(first_child_any_ns(real,"value"), None))
        if qid.startswith("alpha_"):
            alpha = val
        elif qid.startswith("beta_"):
            beta = val

    out["alpha"] = alpha
    out["beta"]  = beta

    # max load from item_data.quantities["math_maximum"]
    max_load = None
    qinfo = item_data.get("quantities", {})
    if "math_maximum" in qinfo:
        max_load = qinfo["math_maximum"].get("value")
    out["max_load"] = max_load

    # build derived table if possible
    if max_load is not None and alpha is not None and beta is not None:
        for pct in [1,25,50,75,100]:
            R = max_load * (pct / 100.0)  # net reading at that % of range
            U = alpha + beta * R
            Urel = (U / R * 100.0) if (R and R != 0) else None
            out["table_rows"].append({
                "pct": pct,
                "R": R,
                "U": U,
                "Urel_percent": Urel
            })

    return out


def extract_minimum_net_weight_example(extras: List[ET.Element]) -> Dict[str, Any]:
    """
    Parse the 2nd basic_contentOutsideDCC block ("Minimum net weight example value")
    which, in the PDF, shows required process accuracy, safety factor, and example min net weight.

    Returns {
      "process_accuracy": float or None,
      "safety_factor": float or None,
      "min_net_weight": float or None,
      "unit": str
    }
    """
    out = {
        "process_accuracy": None,
        "safety_factor": None,
        "min_net_weight": None,
        "unit": "—"
    }

    # choose block WITHOUT <formula> but WITH <list>
    list_block = None
    for ex in extras:
        d = first_child(ex, "data")
        if d is not None and first_child(d, "list") is not None:
            list_block = ex
            break
    if list_block is None:
        return out

    lst = first_child(first_child(list_block,"data"), "list")
    if lst is None:
        return out

    # The list is sequence of named quantities: process accuracy, safety factor, minimum net weight.
    for entry in children(lst, "quantity"):
        # read the name text_en/de just for context (not strictly needed)
        real = first_child_any_ns(entry, "real")
        value = parse_float(text_of(first_child_any_ns(real,"value"), None))
        unit  = text_of(first_child_any_ns(real,"unit"), "—")
        # heuristic classification:
        nm = " ".join(
            text_of(c,"") for c in children(first_child(entry,"name"), "content")
        ).lower()

        if "process accuracy" in nm or "prozessgenau" in nm:
            out["process_accuracy"] = value
        elif "safety factor" in nm or "sicherheitsfaktor" in nm:
            out["safety_factor"] = value
        elif "minimum net weight" in nm or "mindesteinwaage" in nm:
            out["min_net_weight"] = value
            out["unit"] = unit

    return out


# ---------- Markdown rendering ----------

def render_header(core, lab, customer) -> str:
    cert_no = core["certificateIds"][0]["value"] if core["certificateIds"] else "—"
    lab_code = lab["code"]
    cust_lines = customer["name_lines"]
    cust_addr = customer["location"]

    md = []
    md.append(f"# Calibration Certificate / Kalibrierschein")
    md.append("")
    md.append(f"**Certificate ID:** {cert_no}")
    md.append(f"**Laboratory code:** {lab_code}")
    md.append("")
    md.append("## Object / Gegenstand")
    md.append(f"- Object: {cert_no} ({core.get('uniqueIdentifier','—')})")
    md.append(f"- Calibration performed: {core.get('beginPerformanceDate','—')} → {core.get('endPerformanceDate','—')}")
    md.append("")
    md.append("## Customer / Auftraggeber")
    if cust_lines:
        for line in cust_lines:
            md.append(f"- {line}")
    md.append(f"- {cust_addr.get('street','—')}")
    md.append(f"- {cust_addr.get('postCode','—')} {cust_addr.get('city','—')}")
    md.append(f"- {cust_addr.get('countryCode','—')}")
    md.append("")
    md.append("## Calibration Laboratory / Kalibrierlaboratorium")
    md.append(f"- {lab['labName']} ({lab['email']})")
    loc = lab["labLocation"]
    md.append(f"- {loc['street']} {loc['streetNo']}, {loc['postCode']} {loc['city']}, {loc['country']}")
    md.append("")
    md.append("> Traceability: All measurement results stated in this certificate are intended to be ")
    md.append("> metrologically traceable to the SI via national or international standards and ")
    md.append("> evaluated following ISO/IEC 17025 principles (DCC statements & metadata).")
    md.append("")
    return "\n".join(md)


def render_item_section(item_data) -> str:
    qinfo = item_data.get("quantities", {})
    max_load = qinfo.get("math_maximum", {}).get("value")
    min_load = qinfo.get("math_minimum", {}).get("value")
    res_val  = qinfo.get("NAWI_resolutionOfDisplayingDevice", {}).get("value")

    md = []
    md.append("## Calibrated Instrument / Kalibriergegenstand")
    md.append("")
    md.append(f"- Type / Modell: **{item_data.get('model','—')}**")
    md.append(f"- Manufacturer / Hersteller: **{item_data.get('manufacturer','—')}**")
    md.append(f"- Serial no. / Serien-Nr.: {item_data.get('serialNumber','—')}")
    md.append(f"- Customer equipment no. / Prüfmittel-Nr.: {item_data.get('equipmentNumber','—')}")
    md.append("")
    md.append("| Quantity | Value |")
    md.append("|----------|-------|")
    md.append(f"| Minimum load / Mindestlast | {fmt_float(min_load, 6)} g |")
    md.append(f"| Maximum load / Höchstlast (Max) | {fmt_float(max_load, 6)} g |")
    md.append(f"| Display resolution / Teilungswert | {fmt_float(res_val, 6)} g |")
    md.append("")
    return "\n".join(md)

def render_responsible_persons(resps: List[Dict[str, Any]]) -> str:
    md = []
    md.append("## Responsible Persons / Verantwortliche Personen")
    md.append("")
    if not resps:
        md.append("_No responsible persons listed in this certificate._")
        md.append("")
        return "\n".join(md)

    md.append("| Role / Funktion | Name | Main Signer |")
    md.append("|-----------------|------|--------------|")
    for r in resps:
        md.append(f"| {r['role']} | {r['name']} | {r['mainSigner']} |")
    md.append("")
    md.append("> Main Signer = certificate signatory / Unterzeichner des Kalibrierscheins")
    md.append("")
    return "\n".join(md)

def render_environment_section(env) -> str:
    md = []
    md.append("## Environmental & Adjustment Conditions / Umgebungs- & Justierstatus")
    md.append("")
    md.append(f"- Temperature at calibration site / Temperatur am Kalibrierort: "
              f"{fmt_float(env['temperature_value'], 0)} {env['temperature_unit']}")
    md.append(f"- Adjustment status / Justierstatus: {env['adjustment_status']}")
    # comment in both languages, if available
    if env["adjustment_comment_de"] or env["adjustment_comment_en"]:
        md.append("  - " + (env["adjustment_comment_de"] or env["adjustment_comment_en"] or "—"))
    md.append("")
    return "\n".join(md)


def render_methods_section(methods_info) -> str:
    md = []
    md.append("## Calibration Procedure / Kalibrierverfahren")
    md.append("")
    for m in methods_info["methods"]:
        if "calibrationmethod" in (m["refType"] or "").lower():
            md.append(f"- {m['name_de'] or m['name_en'] or 'Method'}")
            md.append(f"  - Norm / Guideline: {m['norm']}")
    md.append("")
    md.append("## Reference Standards / Prüfmittel")
    md.append("")
    # We'll fill this in in the main render once we have measuring equipment list.
    return "\n".join(md)


def render_reference_equipment(me_equip_list) -> str:
    md = []
    md.append("| Equipment / Prüfmittel | Class / Klasse | Identification | Reference |")
    md.append("|------------------------|----------------|---------------|-----------|")
    for eq in me_equip_list:
        ids_join = ", ".join(f"{i['issuer']}: {i['value']}" for i in eq["ids"]) or "—"
        md.append(f"| {eq['name_de']} / {eq['name_en']} "
                  f"| {eq['classID']} "
                  f"| {ids_join} "
                  f"| {eq['reference']} |")
    md.append("")
    return "\n".join(md)


def render_repeatability_tables(rep_blocks: List[Dict[str,Any]]) -> str:
    """
    The PDF shows repeatability at (e.g.) 0.1 g and 5 g with 5 repeated measurements,
    plus sample standard deviation s.

    We'll output one combined table with two side-by-side groups if both blocks exist.
    If only one block exists, we just output that.
    """
    md = []
    md.append("## Repeatability / Wiederholbarkeit")
    md.append("Measured values for repeated loading at different nominal test loads.")
    md.append("")

    if not rep_blocks:
        md.append("_No repeatability data available._")
        md.append("")
        return "\n".join(md)

    if len(rep_blocks) == 1:
        b = rep_blocks[0]
        vals = b["measured_values"]
        md.append(f"**{b['title_de']} / {b['title_en']}**")
        md.append("")
        md.append("| # | Measured value (g) |")
        md.append("|---|--------------------|")
        for idx, v in enumerate(vals, start=1):
            md.append(f"| {idx} | {fmt_float(v, 6)} |")
        md.append(f"| s (Std.dev) | {fmt_float(b['stdev_value'], 7)} |")
        md.append("")
        return "\n".join(md)

    # assume 2 blocks -> side by side
    b1, b2 = rep_blocks[0], rep_blocks[1]
    vals1 = b1["measured_values"]
    vals2 = b2["measured_values"]

    # pad to same length
    Lmax = max(len(vals1), len(vals2))
    while len(vals1) < Lmax:
        vals1.append(None)
    while len(vals2) < Lmax:
        vals2.append(None)

    md.append("| # | {} (g) | # | {} (g) |".format(
        b1["title_en"] or b1["title_de"] or "Block 1",
        b2["title_en"] or b2["title_de"] or "Block 2"
    ))
    md.append("|---|-----------|---|-----------|")
    for i in range(Lmax):
        v1 = fmt_float(vals1[i], 6)
        v2 = fmt_float(vals2[i], 6)
        md.append(f"| {i+1} | {v1} | {i+1} | {v2} |")
    md.append(f"| s | {fmt_float(b1['stdev_value'],7)} | s | {fmt_float(b2['stdev_value'],7)} |")
    md.append("")
    return "\n".join(md)


def render_eccentricity(ecc_data: Dict[str, Any]) -> str:
    md = []
    md.append("## Eccentricity / Außermittige Belastung")
    md.append("")
    if ecc_data["nominal_value"] is not None:
        md.append(f"Test load nominal: {fmt_float(ecc_data['nominal_value'],6)} g")
    if ecc_data["reference_value"] is not None:
        md.append(f"Center indication (Mitte): {fmt_float(ecc_data['reference_value'],6)} g")
    md.append("")
    if not ecc_data["positions"]:
        md.append("_No eccentricity data available._")
        md.append("")
        return "\n".join(md)

    md.append("| Position | Indication / Deviation (g) |")
    md.append("|----------|---------------------------|")
    for pos in ecc_data["positions"]:
        lbl = f"{pos['position_label_de']} / {pos['position_label_en']}"
        md.append(f"| {lbl} | {fmt_float(pos['deviation'], 6)} |")
    md.append("")
    md.append("Maximum deviation from center:")
    md.append(f"- |ΔI|_max = {fmt_float(ecc_data['max_error_value'],6)} g")
    md.append("")
    return "\n".join(md)


def render_error_of_indication(err_data: Dict[str,Any]) -> str:
    md = []
    md.append("## Error of Indication / Abweichung der Anzeige")
    md.append("")
    rows = err_data["rows"]
    if not rows:
        md.append("_No indication error data available._")
        md.append("")
        return "\n".join(md)

    md.append("| L (Nominal load, g) | I (Indication, g) | E = I - L (g) | k | U(E) (g) | U_rel(E) (%) |")
    md.append("|---------------------|------------------|---------------|---|-----------|--------------|")

    for r in rows:
        L   = fmt_float(r["L"],   6)
        I   = fmt_float(r["I"],   6)
        E   = fmt_float(r["E"],   6)
        k   = fmt_float(r["k"],   2)
        U   = fmt_float(r["U"],   6)
        if r["Urel_percent"] is None:
            Urel = "—"
        else:
            # show 0.0042 %
            Urel = "{:.4f} %".format(r["Urel_percent"])
        md.append(f"| {L} | {I} | {E} | {k} | {U} | {Urel} |")

    md.append("")
    md.append("Maximum absolute indication error:")
    md.append(f"- |E|_max = {fmt_float(err_data['max_error_abs'],6)} {err_data['max_error_unit']}")
    md.append("")
    md.append("_U(E) is the expanded measurement uncertainty with coverage factor k (typically k≈2, ~95% coverage)._" )
    md.append("")
    return "\n".join(md)


def render_uncertainty_in_use(unc_use: Dict[str,Any]) -> str:
    md = []
    md.append("## Uncertainty of Measurement in Use / Unsicherheit im Gebrauch der Waage")
    md.append("")
    if unc_use["alpha"] is None or unc_use["beta"] is None:
        md.append("_No 'in-use' uncertainty model available._")
        md.append("")
        return "\n".join(md)

    md.append("Model for the expanded uncertainty of the weighing result in use:")
    md.append("")
    md.append(f"U(W) = {fmt_float(unc_use['alpha'],7)} g + {fmt_float(unc_use['beta'],7)} · R")
    md.append("")
    md.append("where R is the net reading in grams, and U(W) is the expanded uncertainty (k≈2).")
    md.append("")

    if unc_use["table_rows"]:
        md.append("| Load % of Max | Net reading R (g) | U(W) (g) | U_rel(W) (%) |")
        md.append("|---------------|------------------|----------|--------------|")
        for row in unc_use["table_rows"]:
            R    = fmt_float(row["R"],6)
            U    = fmt_float(row["U"],6)
            Urel = f"{row['Urel_percent']*100:.4f} %" if row["Urel_percent"] is not None else "—"
            # CAREFUL: row["Urel_percent"] is already in %, we multiply by 100 only if
            # we accidentally kept it as fraction. We'll keep row["Urel_percent"] AS percent.
            # Let's correct that (see below).
        # We'll re-render properly below:
        md = md[:-3]  # drop last 3 lines to rebuild table with correct math

    # rebuild table correctly
    if unc_use["table_rows"]:
        md.append("| Load % of Max | Net reading R (g) | U(W) (g) | U_rel(W) (%) |")
        md.append("|---------------|------------------|----------|--------------|")
        for row in unc_use["table_rows"]:
            Rval    = fmt_float(row["R"],6)
            Uval    = fmt_float(row["U"],6)
            if row["Urel_percent"] is None:
                Urel = "—"
            else:
                Urel = "{:.4f} %".format(row["Urel_percent"])
            md.append(f"| {row['pct']} % | {Rval} | {Uval} | {Urel} |")
        md.append("")

    return "\n".join(md)


def render_min_net_weight_block(min_block: Dict[str,Any]) -> str:
    md = []
    md.append("## Minimum Net Weight Example / Mindesteinwaage (Beispiel)")
    md.append("")
    if (min_block["process_accuracy"] is None and
        min_block["safety_factor"]   is None and
        min_block["min_net_weight"]  is None):
        md.append("_No example calculation provided._")
        md.append("")
        return "\n".join(md)

    md.append("| Parameter | Value |")
    md.append("|-----------|-------|")
    md.append(f"| Required process accuracy / Geforderte Prozessgenauigkeit | {fmt_float(min_block['process_accuracy'],4)} |")
    md.append(f"| Safety factor / Sicherheitsfaktor | {fmt_float(min_block['safety_factor'],2)} |")
    md.append(f"| Minimum net weight / Mindesteinwaage | {fmt_float(min_block['min_net_weight'],6)} {min_block['unit']} |")
    md.append("")
    return "\n".join(md)


# ---------- glue it together ----------

def _generate_markdown_from_dcc_xml(xml_root: ET.Element) -> str:
    core = extract_core_data(xml_root)
    lab  = extract_lab(xml_root)
    cust = extract_customer(xml_root)
    item = extract_item(xml_root)

    methods_info = extract_methods(xml_root)
    ref_eq_list  = extract_reference_equipment(xml_root)
    resps = extract_responsibles(xml_root)
    mr = get_main_measurement_result(xml_root)

    env_info = extract_environment_and_adjustment(mr) if mr is not None else {}

    # find each measurement sub-result by refType
    results_list = children(first_child(mr, "results"), "result") if mr is not None else []

    rep_blocks = []
    ecc_data   = {}
    err_data   = {}
    extras     = []  # basic_contentOutsideDCC

    for r in results_list:
        rtype = r.attrib.get("refType","")
        if "repeatability" in rtype.lower():
            rep_blocks = extract_repeatability(r)
        elif "eccentricity" in rtype.lower():
            # need position maps, we rebuild them inside extract_eccentricity
            # first build pos_label maps from <text>
            data_el = first_child(r, "data")
            pos_label_map_en = {}
            pos_label_map_de = {}
            txt_el = first_child(data_el, "text") if data_el is not None else None
            for c in children(txt_el, "content"):
                raw = text_of(c, "")
                if ":" in raw:
                    key, val = raw.split(":",1)
                    key = key.strip()
                    val = val.strip()
                    lang = c.attrib.get("lang","").lower()
                    if lang == "en":
                        pos_label_map_en[key] = val
                    elif lang == "de":
                        pos_label_map_de[key] = val
            # then full parse:
            ecc_data = _extract_eccentricity_clean(r,
                                                   pos_label_map_en,
                                                   pos_label_map_de)

        elif "errorofindication" in rtype.lower():
            err_data = extract_error_of_indication(r)

        elif "contentoutsidedcc" in rtype.lower():
            extras.append(r)

    unc_use_block = extract_uncertainty_in_use(extras, item)
    min_weight_block = extract_minimum_net_weight_example(extras)

    # --- Build final Markdown document sections
    parts = []
    parts.append(render_header(core, lab, cust))
    parts.append(render_item_section(item))
    parts.append(render_responsible_persons(resps))    
    parts.append(render_environment_section(env_info))
    parts.append(render_methods_section(methods_info))
    parts.append(render_reference_equipment(ref_eq_list))
    parts.append(render_repeatability_tables(rep_blocks))
    parts.append(render_eccentricity(ecc_data))
    parts.append(render_error_of_indication(err_data))
    parts.append(render_uncertainty_in_use(unc_use_block))
    parts.append(render_min_net_weight_block(min_weight_block))

    return "\n".join(parts).strip() + "\n"

def generate_markdown_from_dcc_xml(xml: str) -> str:
    root = ET.fromstring(xml)
    md = _generate_markdown_from_dcc_xml(root)
    return md

def main(argv=None):
    if argv is None:
        argv = sys.argv
    if len(argv) < 2:
        print("Usage: python dcc_to_markdown.py <dcc.xml>", file=sys.stderr)
        sys.exit(1)

    xml_path = argv[1]
    tree = ET.parse(xml_path)
    root = tree.getroot()

    md = _generate_markdown_from_dcc_xml(root)
    print(md)

if __name__ == "__main__":
    main()