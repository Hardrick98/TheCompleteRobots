import xml.etree.ElementTree as ET

def calcola_lunghezza_totale(urdf_file):
    # Parsing del file URDF
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    
    lunghezze = {}

    # Ciclo su tutti i link nel file URDF
    for link in root.findall(".//link"):
        link_name = link.get("name")
        lunghezza = 0.0

        # Ciclo su tutte le geometrie visuali e di collisione per ogni link
        for visual in link.findall(".//visual"):
            for geometry in visual.findall("geometry"):
                if geometry.tag == "box":
                    # Estrai le dimensioni del box
                    size = geometry.find("box").get("size")
                    x, y, z = map(float, size.split())

                    # Aggiungi la lunghezza lungo l'asse z (supponendo che l'asse z sia la lunghezza)
                    lunghezza += z

                # Se vuoi calcolare anche altre geometrie, aggiungi i controlli per 'cylinder', 'sphere', ecc.

        # Salva la lunghezza totale per il link
        lunghezze[link_name] = lunghezza
    
    return lunghezze

# Esegui il calcolo
urdf_file = "Khr3hv.urdf"  # Sostituisci con il percorso del tuo file URDF
lunghezze_totali = calcola_lunghezza_totale(urdf_file)

# Stampa le lunghezze di ogni link
for link_name, lunghezza in lunghezze_totali.items():
    print(f"Link: {link_name}, Lunghezza: {lunghezza:.3f} metri")
