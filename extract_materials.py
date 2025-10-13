import xml.etree.ElementTree as ET
import json
from pathlib import Path

def extract_materials(urdf_path: str, output_path: str = "materials.json"):
    urdf_path = Path(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # ---- 1️⃣ Estrae materiali globali ----
    global_materials = {}
    for m in root.findall("material"):
        name = m.attrib.get("name")
        color_tag = m.find("color")
        if name and color_tag is not None:
            rgba = list(map(float, color_tag.attrib["rgba"].split()))
            global_materials[name] = rgba

    # ---- 2️⃣ Associa ogni link al suo colore ----
    link_colors = {}

    for link in root.findall("link"):
        link_name = link.attrib["name"]
        visual = link.find("visual")

        if visual is None:
            continue

        material_tag = visual.find("material")

        if material_tag is None:
            continue

        color = None

        # Caso 1: materiale inline nel <visual>
        color_tag = material_tag.find("color")
        if color_tag is not None:
            color = list(map(float, color_tag.attrib["rgba"].split()))

        # Caso 2: riferimento a materiale globale
        elif "name" in material_tag.attrib:
            mat_name = material_tag.attrib["name"]
            color = global_materials.get(mat_name)

        if color is not None:
            link_colors[link_name] = color
        else:
            print(f"⚠️ Nessun colore trovato per link '{link_name}' (materiale: {material_tag.attrib.get('name', 'inline')})")

    # ---- 3️⃣ Salva il dizionario in JSON ----
    with open(output_path, "w") as f:
        json.dump(link_colors, f, indent=4)

    print(f"✅ Salvato dizionario materiali in: {output_path}")
    print(f"   ({len(link_colors)} link con colore estratto)")

if __name__ == "__main__":
    extract_materials("URDF/g1.urdf")