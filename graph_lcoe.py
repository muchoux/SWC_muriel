import os
import pandas as pd
import matplotlib.pyplot as plt


from step_by_step import (
    LCOE, PVI,PVOM
)

df_params = pd.read_csv("parameters/rapport_2025.csv", sep=";", decimal=",")

scenarios = [col for col in df_params.columns if "_" in col]
cities = [sc.split("_")[0] for sc in scenarios]
params_to_have = ["Hopt", "P","Cu", "COM", "PR", "rd", "d", "N", "rom", "T", "Xd", "Nd"]

results = {}
ps_to_plot = {}

for sc in scenarios:
    params = {}
    for x in params_to_have:
        value = df_params.loc[df_params["parameter"] == x, sc].values
        if len(value) > 0: 
            params[x] = float(value[0])
        else:
            params[x] = None 
    
    pvi = PVI(params["P"], params["Cu"])
    pvom = PVOM(pvi,params["COM"]/ 100)
    
    ps_to_plot[sc.split("_")[0]] = float(df_params.loc[df_params["parameter"] == "ps", sc].values[0])
    
    results[sc] = LCOE(pvi, params["Hopt"], params["P"], params["PR"]/ 100, params["rd"]/ 100, params["d"]/ 100, int(params["N"]),params["rom"]/ 100, pvom, params["T"]/ 100, params["Xd"]/ 100, int(params["Nd"]))

df = pd.DataFrame([
    {"City": k.split("_")[0], "Technology": k.split("_")[1], "Value": v}
    for k, v in results.items()
])

df_pivot = df.pivot(index="City", columns="Technology", values="Value")

# --- Plot ---
bar_colors = ["#331921", "#9e011b"]
ax = df_pivot.plot(kind="bar", figsize=(8,6), width=0.7, color=bar_colors)

x_positions = range(len(df_pivot))
y_values = [ps_to_plot[city] for city in df_pivot.index]
line_indicator, = ax.plot(x_positions, y_values, color='red', marker='^', markersize=12, linewidth=2)

# --- Legend for technos ---
leg1 = ax.legend(title="Technologies", fontsize=10, title_fontsize=13, loc="upper right")

# --- legend for ps ---
leg2 = ax.legend([line_indicator], ["ps"], title="ps", fontsize=12, title_fontsize=13)
leg2.set_bbox_to_anchor((0.8, 1))

ax.add_artist(leg1)

plt.title("LCOE per cities", fontsize=18)
plt.xlabel("City", fontsize=14)
plt.ylabel("Price ($/kWh)", fontsize=14)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

output_dir = "results"
png_filename = os.path.join(
    output_dir,
    f"LCOE_bar_plot.png"
)
try : 
    plt.savefig(png_filename)
except:
    print(f"Failed to save png to {png_filename}")
finally:
    print(f"Succesfully saved png to {png_filename}")