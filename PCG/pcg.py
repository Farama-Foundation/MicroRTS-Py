import random
import xml.etree.cElementTree as ET

KEY = 15
SECTIONS = []
UNIT_LOCATION_RECORDS = []
SECTIONS_CHOICES = [0, 1, 2, 3]
BASE_LOCATION_RECORDS = []


def initiate_terrain(root, tag, wallRings):
    terrain = ET.SubElement(root, tag)
    eText = ""
    global UNIT_LOCATION_RECORDS

    def get_obstacle():
        chance = 0.2 * random.random()
        if random.random() < chance:
            return "1"
        else:
            return "0"

    for y in range(height):
        for x in range(width):
            if y in range(0, wallRings) or y in range(height - wallRings, height):
                eText += "1"
                UNIT_LOCATION_RECORDS.append((x, y))
            elif x in range(0, wallRings) or x in range(height - wallRings, height):
                eText += "1"
                UNIT_LOCATION_RECORDS.append((x, y))
            else:
                obstacle = get_obstacle()
                eText += obstacle
                if obstacle == "1":
                    UNIT_LOCATION_RECORDS.append((x, y))

    terrain.text = eText


def initiate_players(root, tag):
    players = ET.SubElement(root, tag)
    ET.SubElement(players, "rts.Player", ID="0", resources="5")
    ET.SubElement(players, "rts.Player", ID="1", resources="5")


def initiate_units(root, tag, wallRings):
    units = ET.SubElement(root, tag)
    int(root.attrib.get("height"))
    int(root.attrib.get("width"))
    initiate_resources(units, "rts.units.Unit")
    initiate_bases(units, "rts.units.Unit")
    initiate_workers(units, "rts.units.Unit")


def initiate_resources(root, tag):
    num_resource = 4

    for i in range(num_resource):
        x, y = get_xy(i)
        ET.SubElement(
            root, tag, type="Resource", ID=get_unique_key(), player="-1", x=str(x), y=str(y), resources="25", hitpoints="1"
        )


def initiate_bases(root, tag):
    num_bases = 2
    index_list = SECTIONS_CHOICES.copy()
    global BASE_LOCATION_RECORDS

    for i in range(num_bases):
        index = random.choice(index_list)
        BASE_LOCATION_RECORDS.append(index)
        index_list.remove(index)
        x, y = get_xy(index)
        ET.SubElement(
            root, tag, type="Base", ID=get_unique_key(), player=str(i % 2), x=str(x), y=str(y), resources="0", hitpoints="10"
        )


def initiate_workers(root, tag):
    num_worker = 2
    for i in range(num_worker):
        x, y = get_xy(BASE_LOCATION_RECORDS[i])
        ET.SubElement(
            root, tag, type="Worker", ID=get_unique_key(), player=str(i % 2), x=str(x), y=str(y), resources="0", hitpoints="1"
        )


def get_unique_key():
    global KEY
    KEY = KEY + 1
    return str(KEY)


def get_xy(index):
    global UNIT_LOCATION_RECORDS

    x, y = random.randint(SECTIONS[index][0][0], SECTIONS[index][0][1]), random.randint(
        SECTIONS[index][1][0], SECTIONS[index][1][1]
    )
    while (x, y) in UNIT_LOCATION_RECORDS:
        x, y = random.randint(SECTIONS[index][0][0], SECTIONS[index][0][1]), random.randint(
            SECTIONS[index][1][0], SECTIONS[index][1][1]
        )
    UNIT_LOCATION_RECORDS.append((x, y))
    return x, y


if __name__ == "__main__":

    root = ET.Element("rts.PhysicalGameState", width="16", height="16")
    height = int(root.attrib.get("height"))
    width = int(root.attrib.get("width"))
    wallRingsLimit = min(height, width) // 2 - 3
    if wallRingsLimit < 0:
        wallRingsLimit = 0
    wallRings = random.randint(0, wallRingsLimit)
    SECTIONS = [
        ((wallRings, (width - 1) // 2), (wallRings, (height - 1) // 2)),
        ((width // 2, (width - 1) - wallRings), (wallRings, (height - 1) // 2)),
        ((wallRings, (width - 1) // 2), (height // 2, (height - 1) - wallRings)),
        ((width // 2, (width - 1) - wallRings), (height // 2, (height - 1) - wallRings)),
    ]

    initiate_terrain(root, "terrain", wallRings)
    initiate_players(root, "players")
    initiate_units(root, "units", wallRings)

    tree = ET.ElementTree(root)
    tree.write("PCG/maps/filename.xml")
