# Procedural Content Generation for Maps

Run this command to get a PCG (Procedural Content Generation) generated map:

```bash
cd PCG
python pcg.py --width 16 --height 16
```

Such a command would generate a map at `maps/filename.xml`. You may use microrts's GUI editor at `gym_microrts/microrts/src/gui/frontend/FrontEnd.java` to visualize the map.

```
bash build.sh
java -cp gym_microrts/microrts/microrts.jar gui.frontend.FrontEnd
```
