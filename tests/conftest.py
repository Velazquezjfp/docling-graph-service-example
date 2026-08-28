import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ONTOLOGY_PATH = ROOT.parent / "user-manual-books" / "handbuch_daten" / "Ontologie" / "ontology.yaml"
ZSD_PDF = ROOT.parent / "user-manual-books" / "handbuch_daten" / "handbuch" / "Betriebshandbuch_ZSD.pdf"
