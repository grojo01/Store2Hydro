import pypsa
import matplotlib.pyplot as plt
import argparse

def plot_pypsa_network(nc_path, 
                       bus_size=30, 
                       line_width=1.5, 
                       title=None,
                       figsize=(10, 8)):
    """
    Lädt ein PyPSA-Netz aus einer .nc-Datei und stellt es grafisch dar.

    Parameter:
        nc_path (str): Pfad zur .nc-Datei.
        bus_size (int): Symbolgröße der Busse.
        line_width (float): Linienbreite der Leitungen.
        title (str): Plot-Titel.
        figsize (tuple): Plotgröße.
    """
    network = pypsa.Network(nc_path)

    plt.figure(figsize=figsize)
    network.plot(
        bus_sizes=bus_size,
        line_widths=line_width
    )

    if title:
        plt.title(title)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PyPSA .nc network")
    parser.add_argument("path", help="Pfad zur .nc-Datei")
    parser.add_argument("--title", help="Plot-Titel", default=None)
    parser.add_argument("--bus-size", type=int, default=30)
    parser.add_argument("--line-width", type=float, default=1.5)

    args = parser.parse_args()

    plot_pypsa_network(
        args.path,
        bus_size=args.bus_size,
        line_width=args.line_width,
        title=args.title
    )