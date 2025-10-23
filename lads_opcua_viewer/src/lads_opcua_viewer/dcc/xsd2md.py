
#!/usr/bin/env python3
"""
xsd2md.py — XML ➜ Markdown renderer driven by XSD (stdlib only).

Usage:
  python xsd2md.py --xsd schema.xsd --xml input.xml [--root ROOTNAME] [--title "Document Title"] > out.md

Goals:
- Derive structure *only* from the XSD; ignore XML bits not declared in the schema.
- Skip images/logos (base64Binary/simple content that looks like images).
- Prefer tables for repeated, record-like elements; otherwise use nested sections and key–value rendering.
- Stdlib-only, so it runs anywhere (including restricted sandboxes). Not a full XSD validator.

Limitations:
- Supports common XSD constructs: global elements, complexType with sequence (and simpleContent), attributes, min/maxOccurs, ref, and local named elements.
- Partial support for 'all' and 'choice' (treated like sequence). Wildcards/any, groups, and advanced facets are not implemented.
"""

from __future__ import annotations
import argparse
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterable, Set

XSD_NS = "http://www.w3.org/2001/XMLSchema"
NSMAP_XSD = {"xs": XSD_NS, "xsd": XSD_NS}

# -------------------------
# Schema model definitions
# -------------------------

@dataclass
class AttributeUse:
    name: str
    type_qname: Optional[Tuple[str,str]]  # (ns, local) or None
    use: str = "optional"  # optional | required | prohibited
    doc: Optional[str] = None

@dataclass
class ElementRef:
    name: Optional[str] = None                   # local name if locally declared
    ref_qname: Optional[Tuple[str,str]] = None   # reference to a global element
    type_qname: Optional[Tuple[str,str]] = None  # reference to a global type
    min_occurs: int = 1
    max_occurs: Optional[int] = 1  # None = unbounded
    doc: Optional[str] = None
    # Inline type details (if present)
    complex_type: Optional['ComplexType'] = None
    simple_type: Optional['SimpleType'] = None

@dataclass
class ComplexType:
    qname: Optional[Tuple[str,str]] = None
    attributes: List[AttributeUse] = field(default_factory=list)
    particles: List[ElementRef] = field(default_factory=list)  # sequence-like order
    simple_content_base: Optional[Tuple[str,str]] = None       # (ns, local) base type if simpleContent
    doc: Optional[str] = None

@dataclass
class SimpleType:
    qname: Optional[Tuple[str,str]] = None
    base_qname: Optional[Tuple[str,str]] = None
    enumerations: List[str] = field(default_factory=list)
    doc: Optional[str] = None

@dataclass
class GlobalElement:
    name: str
    qname: Tuple[str,str]
    type_qname: Optional[Tuple[str,str]] = None
    complex_type: Optional[ComplexType] = None
    simple_type: Optional[SimpleType] = None
    doc: Optional[str] = None

@dataclass
class SchemaModel:
    target_ns: Optional[str]
    elements: Dict[Tuple[str,str], GlobalElement] = field(default_factory=dict)
    complex_types: Dict[Tuple[str,str], ComplexType] = field(default_factory=dict)
    simple_types: Dict[Tuple[str,str], SimpleType] = field(default_factory=dict)
    element_form_qualified: bool = False
    attribute_form_qualified: bool = False
    nsmap: Dict[str,str] = field(default_factory=dict)

# -------------------------
# Utilities
# -------------------------

def _collect_nsmap(root: ET.Element) -> Dict[str,str]:
    ns = {}
    for k,v in root.attrib.items():
        if k.startswith("xmlns:"):
            ns[k.split(":",1)[1]] = v
        elif k == "xmlns":
            ns[""] = v
    # Always include xs/xsd
    ns.setdefault("xs", XSD_NS)
    ns.setdefault("xsd", XSD_NS)
    return ns

def _resolve_qname(token: str, nsmap: Dict[str,str], default_ns: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if token is None:
        return (None, None)
    if ":" in token:
        prefix, local = token.split(":", 1)
        return (nsmap.get(prefix), local)
    else:
        # In XSD attribute QNames (like type="FooType"), no prefix means targetNamespace
        return (default_ns, token)

def _get_text(el: ET.Element) -> Optional[str]:
    if el is None:
        return None
    t = el.text or ""
    t = t.strip()
    return t or None

def _get_doc(el: ET.Element) -> Optional[str]:
    # xs:annotation/xs:documentation
    for ann in el.findall(f"{{{XSD_NS}}}annotation"):
        for doc in ann.findall(f"{{{XSD_NS}}}documentation"):
            txt = "".join(doc.itertext()).strip()
            if txt:
                return txt
    return None

def _attr_int(el: ET.Element, name: str, default: Optional[int]) -> Optional[int]:
    v = el.attrib.get(name)
    if v is None:
        return default
    if v == "unbounded":
        return None
    try:
        return int(v)
    except ValueError:
        return default

def _is_imageish_name(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("image", "logo", "icon", "picture", "img", "thumbnail"))

def _md_escape(text: str) -> str:
    # Minimal Markdown escaping for pipes/backticks/asterisks/underscores
    return (text.replace("|", r"\|")
                .replace("`", r"\`")
                .replace("*", r"\*")
                .replace("_", r"\_")
                .replace("<", "&lt;")
                .replace(">", "&gt;"))

# -------------------------
# XSD Parsing
# -------------------------

def parse_xsd(xsd_path: str) -> SchemaModel:
    tree = ET.parse(xsd_path)
    root = tree.getroot()
    if root.tag != f"{{{XSD_NS}}}schema":
        raise RuntimeError("Root is not xs:schema")

    nsmap = _collect_nsmap(root)
    tns = root.attrib.get("targetNamespace")
    element_form_qualified = (root.attrib.get("elementFormDefault") == "qualified")
    attribute_form_qualified = (root.attrib.get("attributeFormDefault") == "qualified")

    model = SchemaModel(
        target_ns=tns, nsmap=nsmap,
        element_form_qualified=element_form_qualified,
        attribute_form_qualified=attribute_form_qualified,
    )

    # First pass: collect global simple/complex types
    for ct in root.findall(f"{{{XSD_NS}}}complexType"):
        name = ct.attrib.get("name")
        if not name:
            continue
        ctype = _parse_complex_type(ct, nsmap, tns)
        ctype.qname = (tns, name)
        ctype.doc = _get_doc(ct)
        model.complex_types[(tns, name)] = ctype

    for st in root.findall(f"{{{XSD_NS}}}simpleType"):
        name = st.attrib.get("name")
        if not name:
            continue
        stype = _parse_simple_type(st, nsmap, tns)
        stype.qname = (tns, name)
        stype.doc = _get_doc(st)
        model.simple_types[(tns, name)] = stype

    # Second pass: collect global elements
    for ge in root.findall(f"{{{XSD_NS}}}element"):
        name = ge.attrib.get("name")
        if not name:
            continue
        gel = GlobalElement(name=name, qname=(tns, name), doc=_get_doc(ge))
        # Inline type?
        ct = ge.find(f"{{{XSD_NS}}}complexType")
        st = ge.find(f"{{{XSD_NS}}}simpleType")
        t_attr = ge.attrib.get("type")
        if t_attr:
            gel.type_qname = _resolve_qname(t_attr, nsmap, tns)
        elif ct is not None:
            gel.complex_type = _parse_complex_type(ct, nsmap, tns)
        elif st is not None:
            gel.simple_type = _parse_simple_type(st, nsmap, tns)
        model.elements[gel.qname] = gel

    return model

def _parse_complex_type(ct: ET.Element, nsmap: Dict[str,str], tns: Optional[str]) -> ComplexType:
    ctype = ComplexType()
    ctype.doc = _get_doc(ct)

    # simpleContent?
    sc = ct.find(f"{{{XSD_NS}}}simpleContent")
    if sc is not None:
        ext = sc.find(f"{{{XSD_NS}}}extension")
        if ext is not None and "base" in ext.attrib:
            ctype.simple_content_base = _resolve_qname(ext.attrib["base"], nsmap, tns)
            # attributes in extension
            for at in ext.findall(f"{{{XSD_NS}}}attribute"):
                ctype.attributes.append(_parse_attribute(at, nsmap, tns))
        # If restriction present, we ignore for now
    else:
        # sequence / all / choice treated similarly as ordered particles
        particles_parent = None
        for tag in ("sequence", "all", "choice"):
            cand = ct.find(f"{{{XSD_NS}}}{tag}")
            if cand is not None:
                particles_parent = cand
                break
        if particles_parent is not None:
            for el in particles_parent.findall(f"{{{XSD_NS}}}element"):
                ctype.particles.append(_parse_element_ref(el, nsmap, tns))
        # attributes
        for at in ct.findall(f"{{{XSD_NS}}}attribute"):
            ctype.attributes.append(_parse_attribute(at, nsmap, tns))

    return ctype

def _parse_simple_type(st: ET.Element, nsmap: Dict[str,str], tns: Optional[str]) -> SimpleType:
    stype = SimpleType()
    stype.doc = _get_doc(st)
    # restriction base
    restr = st.find(f"{{{XSD_NS}}}restriction")
    if restr is not None and "base" in restr.attrib:
        stype.base_qname = _resolve_qname(restr.attrib["base"], nsmap, tns)
        for enum in restr.findall(f"{{{XSD_NS}}}enumeration"):
            val = enum.attrib.get("value")
            if val is not None:
                stype.enumerations.append(val)
    return stype

def _parse_attribute(at: ET.Element, nsmap: Dict[str,str], tns: Optional[str]) -> AttributeUse:
    name = at.attrib.get("name")
    use = at.attrib.get("use", "optional")
    t_attr = at.attrib.get("type")
    doc = _get_doc(at)
    type_qname = _resolve_qname(t_attr, nsmap, tns) if t_attr else None
    return AttributeUse(name=name, type_qname=type_qname, use=use, doc=doc)

def _parse_element_ref(el: ET.Element, nsmap: Dict[str,str], tns: Optional[str]) -> ElementRef:
    er = ElementRef()
    er.name = el.attrib.get("name")
    if "ref" in el.attrib:
        er.ref_qname = _resolve_qname(el.attrib["ref"], nsmap, tns)
    if "type" in el.attrib:
        er.type_qname = _resolve_qname(el.attrib["type"], nsmap, tns)
    er.min_occurs = _attr_int(el, "minOccurs", 1) or 0
    er.max_occurs = _attr_int(el, "maxOccurs", 1)
    er.doc = _get_doc(el)

    # inline type
    ct = el.find(f"{{{XSD_NS}}}complexType")
    st = el.find(f"{{{XSD_NS}}}simpleType")
    if ct is not None:
        er.complex_type = _parse_complex_type(ct, nsmap, tns)
    if st is not None:
        er.simple_type = _parse_simple_type(st, nsmap, tns)
    return er

# -------------------------
# Rendering
# -------------------------

def _typename(qname: Optional[Tuple[str,str]]) -> Optional[str]:
    if not qname:
        return None
    ns, loc = qname
    if ns == XSD_NS:
        return f"xs:{loc}"
    if ns:
        return f"{{{ns}}}{loc}"
    return loc

def _is_base64(qname: Optional[Tuple[str,str]]) -> bool:
    return qname == (XSD_NS, "base64Binary")

def _is_simple_builtin(qname: Optional[Tuple[str,str]]) -> bool:
    return qname and qname[0] == XSD_NS and qname[1] not in ("anyType", "anySimpleType")

def _element_schema(model: SchemaModel, parent_ct: Optional[ComplexType], xml_tag: str) -> Optional[ElementRef]:
    # xml_tag like "{ns}local", need to match by local and ns against parent's particles
    xml_ns, xml_local = _split_tag(xml_tag)
    if parent_ct:
        for p in parent_ct.particles:
            # resolve referenced or local name and compare
            pname = p.name
            pns = model.target_ns
            if p.ref_qname:
                pns, pname = p.ref_qname
            if pname == xml_local and (pns == xml_ns or (xml_ns is None and pns is None)):
                return p
    return None

def _split_tag(tag: str) -> Tuple[Optional[str], str]:
    if tag.startswith("{"):
        ns, local = tag[1:].split("}")
        return ns, local
    else:
        return None, tag

def _find_global_element(model: SchemaModel, xml_root: ET.Element, prefer_name: Optional[str]=None) -> Optional[GlobalElement]:
    rns, rlocal = _split_tag(xml_root.tag)
    # First try exact (ns,local)
    cand = model.elements.get((rns, rlocal))
    if cand:
        return cand
    # Try any element with same local name if prefer_name is given
    if prefer_name:
        for (ns, ln), ge in model.elements.items():
            if ln == prefer_name:
                return ge
    # Fallback: if only one global element, use it
    if len(model.elements) == 1:
        return next(iter(model.elements.values()))
    return None

def _is_leaf_simple(model: SchemaModel, er: ElementRef) -> bool:
    # A leaf is: simpleType, or complexType with simpleContent, and no child particles
    if er.simple_type is not None:
        return True
    if er.type_qname:
        # Resolve known simple type
        if _is_simple_builtin(er.type_qname) or er.type_qname in model.simple_types:
            return True
        # Complex type name?
        if er.type_qname in model.complex_types:
            ct = model.complex_types[er.type_qname]
            if ct.simple_content_base is not None:
                return True
            if not ct.particles:
                # No children -> could be attributes only
                return True
    if er.complex_type:
        if er.complex_type.simple_content_base is not None:
            return True
        if not er.complex_type.particles:
            return True
    return False

def _collect_record_fields(model: SchemaModel, er: ElementRef) -> List[Tuple[str,str]]:
    """Return list of (kind,name) where kind in {'attr','child','text'} for a record-like element.
    """
    fields: List[Tuple[str,str]] = []
    # Attributes from type
    def add_attrs(ct: ComplexType):
        for at in ct.attributes:
            if at.name and not _is_imageish_name(at.name):
                fields.append(("attr", at.name))
    # Children
    def add_child_names(ct: ComplexType):
        for p in ct.particles:
            name = p.name or (p.ref_qname[1] if p.ref_qname else None)
            if not name:
                continue
            if _is_leaf_simple(model, p) and not _is_imageish_name(name):
                fields.append(("child", name))
    # Inline or named type
    if er.complex_type:
        ct = er.complex_type
        add_attrs(ct)
        add_child_names(ct)
        # simpleContent text
        if ct.simple_content_base is not None and not _is_base64(ct.simple_content_base):
            fields.append(("text", "#text"))
    elif er.type_qname:
        # Referenced type
        if er.type_qname in model.complex_types:
            ct = model.complex_types[er.type_qname]
            add_attrs(ct)
            add_child_names(ct)
            if ct.simple_content_base is not None and not _is_base64(ct.simple_content_base):
                fields.append(("text", "#text"))
        elif _is_simple_builtin(er.type_qname) or er.type_qname in model.simple_types:
            fields.append(("text", "#text"))
    elif er.simple_type is not None:
        fields.append(("text", "#text"))
    # Deduplicate keeping order
    seen = set()
    out = []
    for k,n in fields:
        key = (k,n)
        if key not in seen:
            out.append((k,n))
            seen.add(key)
    return out

def _recordlike(model: SchemaModel, er: ElementRef) -> bool:
    # Consider record-like if all particles (or the er itself) are leaf simples
    # i.e., child elements are leaves; Also avoid base64
    # If the element itself is simple, it's a scalar, not a record container.
    # We're deciding if repeated er elements should be tabular rows.
    # Criterion: er is complex, and all child particles (if any) are leaf simples.
    ct = None
    if er.complex_type:
        ct = er.complex_type
    elif er.type_qname and er.type_qname in model.complex_types:
        ct = model.complex_types[er.type_qname]
    if not ct:
        return False
    # Has at least one leaf child or text content
    leafy = True
    has_any_field = False
    for p in ct.particles:
        if _is_imageish_name(p.name or (p.ref_qname[1] if p.ref_qname else "")):
            continue
        if not _is_leaf_simple(model, p):
            leafy = False
            break
        has_any_field = True
    # also consider simpleContent text as a field
    if ct.simple_content_base is not None and not _is_base64(ct.simple_content_base):
        has_any_field = True
    return leafy and has_any_field

def render_markdown(model: SchemaModel, xml_path: str, title: Optional[str]=None, root_name: Optional[str]=None) -> str:
    xtree = ET.parse(xml_path)
    xroot = xtree.getroot()

    ge = _find_global_element(model, xroot, prefer_name=root_name)
    if not ge:
        raise RuntimeError("Could not map XML root to a global element in the XSD. Use --root to hint.")

    # Heading
    doc_title = title or ge.name
    out_lines: List[str] = [f"# {doc_title}"]

    # Render the root content
    _render_element(out_lines, model, ge, xroot, level=2)

    return "\n".join(out_lines).rstrip() + "\n"

def _render_element(out: List[str], model: SchemaModel, ge_or_er, xel: ET.Element, level: int):
    """Render a global element (ge) or a child element ref (er) against the XML element xel."""
    if isinstance(ge_or_er, GlobalElement):
        # Determine type
        ct = None
        st = None
        if ge_or_er.complex_type:
            ct = ge_or_er.complex_type
        elif ge_or_er.type_qname and ge_or_er.type_qname in model.complex_types:
            ct = model.complex_types[ge_or_er.type_qname]
        elif ge_or_er.simple_type or (ge_or_er.type_qname and (ge_or_er.type_qname in model.simple_types or _is_simple_builtin(ge_or_er.type_qname))):
            st = ge_or_er.simple_type
        # Render
        _render_section_header(out, ge_or_er.name, level-1)  # one level up to keep root prominent
        if ct:
            _render_complex_content(out, model, ct, xel, level)
        else:
            txt = (xel.text or "").strip()
            if txt:
                out.append(txt)
        return

    # It's an ElementRef for a child
    er: ElementRef = ge_or_er
    name = er.name or (er.ref_qname[1] if er.ref_qname else xel.tag.split('}',1)[-1])
    if _is_imageish_name(name):
        return
    # Skip base64-like
    if er.simple_type is None and er.type_qname is not None and _is_base64(er.type_qname):
        return
    if er.complex_type and er.complex_type.simple_content_base and _is_base64(er.complex_type.simple_content_base):
        return

    # Determine type for child
    ct = None
    if er.complex_type:
        ct = er.complex_type
    elif er.type_qname and er.type_qname in model.complex_types:
        ct = model.complex_types[er.type_qname]

    if ct:
        _render_section_header(out, name, level)
        _render_complex_content(out, model, ct, xel, level+1)
    else:
        # simple content (with possible attributes)
        attrs = _collect_attributes(ct, er, model, xel)
        txt = (xel.text or "").strip()
        if txt or attrs:
            # Single-line key-value with label
            line = f"**{_md_escape(name)}:** "
            if txt:
                line += _md_escape(txt)
            if attrs:
                parts = [f"{k}={_md_escape(v)}" for k,v in attrs.items()]
                line += "  " + "  ".join(parts)
            out.append(line)

def _collect_attributes(ct: Optional[ComplexType], er: ElementRef, model: SchemaModel, xel: ET.Element) -> OrderedDict:
    attrs = OrderedDict()
    # Allowed attributes come from ct if available
    def add_attr(aname: str, value: Optional[str]):
        if value is None or value == "":
            return
        if _is_imageish_name(aname):
            return
        attrs[aname] = value

    if ct:
        for at in ct.attributes:
            if not at.name:
                continue
            v = xel.attrib.get(at.name)
            add_attr(at.name, v)
    else:
        # If we don't know attributes from schema (e.g., simpleType), don't add any — schema-driven only.
        pass
    return attrs

def _render_complex_content(out: List[str], model: SchemaModel, ct: ComplexType, xel: ET.Element, level: int):
    # Attributes (key-values at top)
    attrs = OrderedDict()
    for at in ct.attributes:
        if not at.name or _is_imageish_name(at.name):
            continue
        v = xel.attrib.get(at.name)
        if v:
            attrs[at.name] = v
    if attrs:
        for k,v in attrs.items():
            out.append(f"**{_md_escape(k)}:** {_md_escape(v)}")

    # simpleContent text
    if ct.simple_content_base is not None and not _is_base64(ct.simple_content_base):
        txt = (xel.text or "").strip()
        if txt:
            out.append(_md_escape(txt))

    # Children: group by declared particles
    # Build mapping from particle to matching XML children
    children_by_particle: List[Tuple[ElementRef, List[ET.Element]]] = []
    for p in ct.particles:
        # collect children that match this particle name (respect ns)
        pname = p.name or (p.ref_qname[1] if p.ref_qname else None)
        pns = (p.ref_qname[0] if p.ref_qname else model.target_ns)
        if not pname:
            continue
        matches = []
        for child in xel:
            cns, clocal = _split_tag(child.tag)
            if clocal == pname and (cns == pns or pns is None):
                matches.append(child)
        if matches:
            children_by_particle.append((p, matches))

    # Render each group; tables for repeated record-like particles
    for p, items in children_by_particle:
        # Skip image-ish particles entirely
        pname = p.name or (p.ref_qname[1] if p.ref_qname else "")
        if _is_imageish_name(pname):
            continue
        if p.max_occurs is None or (p.max_occurs and p.max_occurs > 1) or len(items) > 1:
            if _recordlike(model, p):
                _render_table(out, model, p, items, level)
                continue
        # Otherwise render each item as nested section
        for it in items:
            _render_element(out, model, p, it, level)

def _render_section_header(out: List[str], title: str, level: int):
    level = max(2, min(6, level))
    out.append(f"{'#'*level} {_md_escape(title)}")

def _extract_field_value(kind: str, name: str, model: SchemaModel, er: ElementRef, item: ET.Element) -> str:
    if kind == "attr":
        return (item.attrib.get(name) or "").strip()
    if kind == "child":
        # find first child with that local name in target ns
        for ch in item:
            _, loc = _split_tag(ch.tag)
            if loc == name:
                # text only (leaf)
                return (ch.text or "").strip()
        return ""
    if kind == "text":
        return (item.text or "").strip()
    return ""

def _render_table(out: List[str], model: SchemaModel, er: ElementRef, items: List[ET.Element], level: int):
    # Determine columns from schema (attrs + child simple leaves + text for simpleContent)
    cols = _collect_record_fields(model, er)
    if not cols:
        # Fallback: list rendering
        for it in items:
            _render_element(out, model, er, it, level)
        return

    header = [n if k != "text" else "value" for (k,n) in cols]
    out.append("")
    out.append("| " + " | ".join(_md_escape(h) for h in header) + " |")
    out.append("| " + " | ".join("---" for _ in header) + " |")
    for it in items:
        row_vals = []
        for kind, name in cols:
            val = _extract_field_value(kind, name, model, er, it)
            row_vals.append(_md_escape(val) if val is not None else "")
        out.append("| " + " | ".join(row_vals) + " |")
    out.append("")

# -------------------------
# CLI
# -------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Render XML (validated by XSD) to Markdown.")
    ap.add_argument("--xsd", required=True, help="Path to XSD schema file")
    ap.add_argument("--xml", required=True, help="Path to XML instance document")
    ap.add_argument("--root", help="Root element local name hint, if mapping is ambiguous")
    ap.add_argument("--title", help="Override document title (Markdown H1)")
    args = ap.parse_args(argv)

    try:
        model = parse_xsd(args.xsd)
    except Exception as e:
        print(f"ERROR parsing XSD: {e}", file=sys.stderr)
        return 2

    try:
        md = render_markdown(model, args.xml, title=args.title, root_name=args.root)
    except Exception as e:
        print(f"ERROR rendering XML: {e}", file=sys.stderr)
        return 3

    sys.stdout.write(md)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
