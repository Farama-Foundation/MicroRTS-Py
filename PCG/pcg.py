import xml.etree.cElementTree as ET
import random

key = 15
sections = []


def initiateTerrain(root, tag, wallRings):
    terrain = ET.SubElement(root, tag)
    eText = ''

    def getObstacles():
        chance = 0.2 * random.random()
        if random.random() < chance:
            return '1'
        else:
            return '0'

    for i in range(height):
        for y in range(width):
            if i in range(0, wallRings) or i in range(height - wallRings, height):
                eText += '1'
            elif y in range(0, wallRings) or y in range(height - wallRings, height):
                eText += '1'
            else:
                eText += getObstacles()
    terrain.text = eText


def initiatePlayers(root, tag):
    players = ET.SubElement(root, tag)
    ET.SubElement(players, "rts.Player", ID="0", resources="5")
    ET.SubElement(players, "rts.Player", ID="1", resources="5")


def initiateUnits(root, tag, wallRings):
    units = ET.SubElement(root, tag)
    height = int(root.attrib.get('height'))
    width = int(root.attrib.get('width'))
    initiateResources(units, "rts.units.Unit", wallRings, height, width)
    initiateBases(units, "rts.units.Unit", wallRings, height, width)
    initiateWorkers(units, "rts.units.Unit", wallRings, height, width)


def initiateResources(root, tag, wallRings, height, width):
    resourcesLimit = 4
    num_resource = 4

    for i in range(num_resource):
        x, y = get_xy(i)
        ET.SubElement(root, tag, type="Resource", ID=get_unique_key(), player="-1",
                      x=str(x), y=str(y), resources="25", hitpoints="1")


def initiateBases(root, tag, wallRings, height, width):
    base_limit = 2
    num_bases = 2
    for i in range(num_bases):
        x, y = get_xy(random.randint(0, 3))
        ET.SubElement(root, tag, type="Base", ID=get_unique_key(),
                      player=str(i % 2), x=str(x), y=str(y), resources="0", hitpoints="10")


def initiateWorkers(root, tag, wallRings, height, width):
    worker_limit = 2
    num_worker = 2
    for i in range(num_worker):
        x, y = get_xy(random.randint(0, 3))
        ET.SubElement(root, tag, type="Worker", ID=get_unique_key(), player=str(i % 2),
                      x=str(x), y=str(y), resources="0", hitpoints="1")


def get_unique_key():
    global key
    key = key + 1
    return str(key)


def get_xy(index):
    return random.randint(sections[index][0][0], sections[index][0][1]), random.randint(sections[index][1][0], sections[index][1][1])


if __name__ == "__main__":

    root = ET.Element("rts.PhysicalGameState", width="16", height="16")
    height = int(root.attrib.get('height'))
    width = int(root.attrib.get('width'))
    wallRingsLimit = min(height, width) // 2 - 3
    if wallRingsLimit < 0:
        wallRingsLimit = 0
    wallRings = random.randint(0, wallRingsLimit)
    sections = [((wallRings, (width-1)//2), (wallRings, (height-1)//2)),  (
        (width//2, (width-1)-wallRings), (wallRings, (height-1)//2)),  ((wallRings, (width-1)//2), (height//2, (height-1)-wallRings)), ((width//2, (width-1)-wallRings), (height//2, (height-1)-wallRings))]

    initiateTerrain(root, "terrain", wallRings)
    initiatePlayers(root, "players")
    initiateUnits(root, "units", wallRings)

    tree = ET.ElementTree(root)
    tree.write("PCG/maps/filename.xml")
