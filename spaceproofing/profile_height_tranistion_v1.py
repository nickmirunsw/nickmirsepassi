import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import LineString

# from shapely.geometry import MultiLineString
# import NumPy

# Read the Excel file
df = pd.read_excel("ps04.xlsx")

# Separate the data into blocks based on the empty rows
blocks = []
current_block = []

for index, row in df.iterrows():
    if pd.notna(row["X"]) and pd.notna(row["Y"]):
        current_block.append((row["X"], row["Y"]))
    elif current_block:
        blocks.append(current_block)
        current_block = []

# Add the last block if it's not empty
if current_block:
    blocks.append(current_block)

# Create a plot with solid lines for each block
fig, ax = plt.subplots()
for block in blocks:
    x, y = zip(*block)
    line = LineString(block)
    ax.plot(
        *line.xy, linestyle="-", marker=None, color="b"
    )  # Solid line without markers

# Check for intersections between blocks
intersections = []
for i in range(len(blocks)):
    for j in range(i + 1, len(blocks)):
        if LineString(blocks[i]).intersects(LineString(blocks[j])):
            intersection = LineString(blocks[i]).intersection(LineString(blocks[j]))
            if intersection.geom_type == "Point":
                x, y = intersection.xy
                intersections.append((x[0], y[0]))

# Show intersections on the plot
if intersections:
    ix, iy = zip(*intersections)
    ax.scatter(ix, iy, color="red", marker="x", label="Intersections")
    for x, y in zip(ix, iy):
        ax.annotate(
            f"({x:.3f}, {y:.3f})",
            (x, y),
            textcoords="offset points",
            xytext=(10, 15),
            ha="center",
        )

# Set plot labels and legend
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()

# Show the plot
plt.show()
