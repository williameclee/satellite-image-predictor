import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Define ordered RGB colors for classes 1 to 10
landcover_colours = [
    "#000000",  # 00: No data
    "#1f78b4",  # 01: Water
    "#ffffff",  # 02: Snow/Ice
    "#b4b4b4",  # 03: Bare
    "#228B22",  # 04: Tree cover
    "#9ACD32",  # 05: Shrubland
    "#ADFF2F",  # 06: Grassland
    "#00CED1",  # 07: Wetlands
    "#DAA520",  # 08: Cropland
    "#E31A1C",  # 09: Built-up
    "#FAE6A0",  # 10: Moss & Lichen
]

landcover = ListedColormap(landcover_colours)
landcovernorm = BoundaryNorm(
    range(0, 12), landcover.N
)  # class values from 1 to 10 (11 = exclusive)

# plt.imshow(landcover_array, cmap=lc_cmap, norm=lc_norm)
