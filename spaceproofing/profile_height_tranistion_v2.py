import pandas as pd

# import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString


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
                # Format the coordinates to four decimal points
                x_formatted = "{:.4f}".format(x[0])
                y_formatted = "{:.4f}".format(y[0])
                intersections.append((float(x_formatted), float(y_formatted)))

# Show intersections on the plot with numbered labels
if intersections:
    ix, iy = zip(*intersections)
    ax.scatter(ix, iy, color="red", marker="x", label="Intersections")
    for k, (x, y) in enumerate(zip(ix, iy), start=1):
        ax.annotate(
            f"{k}",
            (x, y),
            textcoords="offset points",
            xytext=(10, 15),
            ha="center",
            fontsize=8,
        )

    # Create a DataFrame for intersections with formatted coordinates
    intersections_df = pd.DataFrame(
        {
            "Intersection": range(1, len(intersections) + 1),
            "X": ["{:.4f}".format(x) for x in ix],
            "Y": ["{:.4f}".format(y) for y in iy],
        }
    )
    intersections_df.to_excel("intersections_coordinates.xlsx", index=False)

# Set plot labels and legend
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()

# Show the plot
plt.show()
