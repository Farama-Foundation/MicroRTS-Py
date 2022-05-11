import argparse
import os
import random
import time
import xml.etree.cElementTree as ET


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=16,help='the width of the map')
    parser.add_argument('--height', type=int, default=16,help='the height of the map')
    parser.add_argument('--num-maps', type=int, default=200, help='the number of the maps to generate')
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    args = parser.parse_args()
    # fmt: on

    if not args.seed:
        args.seed = int(time.time())
    return args


class PCG:
    def __init__(
        self,
        width=16,
        height=16,
        key=15,
        unit_location_records=[],
        sections_choices=[0, 1, 2, 3],
        base_location_records=[],
        num_maps=200,
    ):
        self.height = height
        self.width = width
        self.wallRingsLimit = min(height, width) // 2 - 3
        if self.wallRingsLimit < 0:
            self.wallRingsLimit = 0
        self.wallRings = random.randint(0, self.wallRingsLimit)
        self.key = key
        self.sections = [
            ((self.wallRings, (width - 1) // 2), (self.wallRings, (height - 1) // 2)),
            ((width // 2, (width - 1) - self.wallRings), (self.wallRings, (height - 1) // 2)),
            ((self.wallRings, (width - 1) // 2), (height // 2, (height - 1) - self.wallRings)),
            ((width // 2, (width - 1) - self.wallRings), (height // 2, (height - 1) - self.wallRings)),
        ]
        self.unit_location_records = unit_location_records
        self.sections_choices = sections_choices
        self.base_location_records = base_location_records
        self.num_maps = num_maps

    def initiate_terrain(self, root, tag, wallRings):
        terrain = ET.SubElement(root, tag)
        eText = ""

        def get_obstacle():
            chance = 0.2 * random.random()
            if random.random() < chance:
                return "1"
            else:
                return "0"

        for y in range(self.height):
            for x in range(self.width):
                if y in range(0, wallRings) or y in range(self.height - wallRings, self.height):
                    eText += "1"
                    self.unit_location_records.append((x, y))
                elif x in range(0, wallRings) or x in range(self.height - wallRings, self.height):
                    eText += "1"
                    self.unit_location_records.append((x, y))
                else:
                    obstacle = get_obstacle()
                    eText += obstacle
                    if obstacle == "1":
                        self.unit_location_records.append((x, y))

        terrain.text = eText

    def initiate_players(self, root, tag):
        players = ET.SubElement(root, tag)
        ET.SubElement(players, "rts.Player", ID="0", resources="5")
        ET.SubElement(players, "rts.Player", ID="1", resources="5")

    def initiate_units(self, root, tag):
        units = ET.SubElement(root, tag)
        int(root.attrib.get("height"))
        int(root.attrib.get("width"))
        self.initiate_resources(units, "rts.units.Unit")
        self.initiate_bases(units, "rts.units.Unit")
        self.initiate_workers(units, "rts.units.Unit")

    def initiate_resources(self, root, tag):
        num_resource = 4

        for i in range(num_resource):
            x, y = self.get_xy(i)
            ET.SubElement(
                root,
                tag,
                type="Resource",
                ID=self.get_unique_key(),
                player="-1",
                x=str(x),
                y=str(y),
                resources="25",
                hitpoints="1",
            )

    def initiate_bases(self, root, tag):
        num_bases = 2
        index_list = self.sections_choices.copy()

        for i in range(num_bases):
            index = random.choice(index_list)
            self.base_location_records.append(index)
            index_list.remove(index)
            x, y = self.get_xy(index)
            ET.SubElement(
                root,
                tag,
                type="Base",
                ID=self.get_unique_key(),
                player=str(i % 2),
                x=str(x),
                y=str(y),
                resources="0",
                hitpoints="10",
            )

    def initiate_workers(self, root, tag):
        num_worker = 2
        for i in range(num_worker):
            x, y = self.get_xy(self.base_location_records[i])
            ET.SubElement(
                root,
                tag,
                type="Worker",
                ID=self.get_unique_key(),
                player=str(i % 2),
                x=str(x),
                y=str(y),
                resources="0",
                hitpoints="1",
            )

    def get_unique_key(self):
        self.key = self.key + 1
        return str(self.key)

    def get_xy(self, index):
        x, y = random.randint(self.sections[index][0][0], self.sections[index][0][1]), random.randint(
            self.sections[index][1][0], self.sections[index][1][1]
        )
        while (x, y) in self.unit_location_records:
            x, y = random.randint(self.sections[index][0][0], self.sections[index][0][1]), random.randint(
                self.sections[index][1][0], self.sections[index][1][1]
            )
        self.unit_location_records.append((x, y))
        return x, y

    def get_map(self, mapKey):
        root = ET.Element("rts.PhysicalGameState", width=str(self.width), height=str(self.height))
        self.initiate_terrain(root, "terrain", self.wallRings)
        self.initiate_players(root, "players")
        self.initiate_units(root, "units")
        tree = ET.ElementTree(root)
        tree.write(os.path.join("PCG/maps/", "pcg_map" + "_" + str(mapKey) + ".xml"))
        self.reset()

        return tree

    def reset(self):
        self.unit_location_records = []
        self.base_location_records = []

    def get_maps(self):
        for i in range(self.num_maps):
            self.get_map(i)


if __name__ == "__main__":
    if not os.path.exists("PCG/maps"):
        os.makedirs("PCG/maps")
    args = parse_args()
    random.seed(args.seed)
    pcg = PCG(width=args.width, height=args.height, num_maps=args.num_maps)
    pcg.get_maps()
