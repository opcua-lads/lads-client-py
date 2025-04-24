import os
import csv
from typing import Dict, Optional, List

class DictionaryEntry:
    def __init__(self, termIRI: str, prefLabel: str, altLabels: str = "",
                 domain: str = "", termType: str = "", definition: str = "",
                 scopeNote: str = "", source: str = "", examples: str = "",
                 parents: str = ""):
        self.termIRI = termIRI
        self.prefLabel = prefLabel
        self.altLabels = altLabels
        self.domain = domain
        self.termType = termType
        self.definition = definition
        self.scopeNote = scopeNote
        self.sources: List[str] = []
        if source:
            self.sources.append(source)
        self.examples = examples
        self.parents = parents

    def add_source(self, source: str) -> None:
        """Adds a new source if non-empty and not already present."""
        if source and source not in self.sources:
            self.sources.append(source)

    def to_dict(self) -> Dict:
        """Returns a dictionary representation of the entry."""
        return {
            'termIRI': self.termIRI,
            'prefLabel': self.prefLabel,
            'altLabels': self.altLabels,
            'domain': self.domain,
            'termType': self.termType,
            'definition': self.definition,
            'scopeNote': self.scopeNote,
            'sources': self.sources,
            'examples': self.examples,
            'parents': self.parents
        }

    def __repr__(self) -> str:
        return str(self.to_dict())


def load_dictionary_csv(file_path: str) -> Dict[str, DictionaryEntry]:
    """
    Loads the CSV file and returns a dictionary mapping termIRI to a DictionaryEntry instance.
    If the same termIRI appears in multiple rows, the corresponding source values are aggregated.
    """
    entries: Dict[str, DictionaryEntry] = {}
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            termIRI = row.get('TermIRI', '').strip()
            if not termIRI:
                continue

            # Extract other fields, defaulting to an empty string if not present.
            prefLabel   = row.get('PrefLabel', '').strip()
            altLabels   = row.get('AltLabels', '').strip()
            domain      = row.get('Domain', '').strip()
            termType    = row.get('TermType', '').strip()
            definition  = row.get('Definition', '').strip()
            scopeNote   = row.get('ScopeNote', '').strip()
            source      = row.get('Source', '').strip()
            examples    = row.get('Examples', '').strip()
            parents     = row.get('Parents', '').strip()

            if termIRI in entries:
                entries[termIRI].add_source(source)
            else:
                entry = DictionaryEntry(
                    termIRI=termIRI,
                    prefLabel=prefLabel,
                    altLabels=altLabels,
                    domain=domain,
                    termType=termType,
                    definition=definition,
                    scopeNote=scopeNote,
                    source=source,
                    examples=examples,
                    parents=parents
                )
                entries[termIRI] = entry
    return entries


# ----- Module Initialization -----
# Determine the CSV file path assuming the following structure:
# project_root/
#   data/AFO_Dictionary-2025_03.csv
#   src/lads_afo.py
_csv_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'AFO_Dictionary-2025_03.csv')
try:
    _DICTIONARY_ENTRIES: Dict[str, DictionaryEntry] = load_dictionary_csv(_csv_file_path)
    print(f"Loaded {_DICTIONARY_ENTRIES.__len__()} dictionary entries from CSV.")
except Exception as e:
    print(f"Error loading dictionary CSV: {e}")
    _DICTIONARY_ENTRIES = {}


# ----- Public API Functions -----
def get_entry(termIRI: str) -> Optional[DictionaryEntry]:
    """
    Returns the DictionaryEntry for the given termIRI.
    If the termIRI is not found, returns None.
    """
    return _DICTIONARY_ENTRIES.get(termIRI)

def get_all_entries() -> List[DictionaryEntry]:
    """
    Returns a list of all DictionaryEntry instances.
    """
    return list(_DICTIONARY_ENTRIES.values())

