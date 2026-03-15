"""
Interactive map generation for urban growth analysis.

Uses Folium and/or Geemap to visualize:
    - Satellite RGB imagery
    - Classification maps (urban / semi-urban / non-urban)
    - Change detection heatmaps
    - GIS overlays (roads, boundaries)

Output saved as HTML interactive maps.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import folium
    from folium import plugins as folium_plugins
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    logger.warning("folium not installed. Install: pip install folium")

try:
    import geemap
    GEEMAP_AVAILABLE = True
except ImportError:
    GEEMAP_AVAILABLE = False

# Vijayawada center coordinates
VIJAYAWADA_CENTER = [16.5062, 80.6480]

CLASS_COLORS = {
    0: "#2ECC71",   # Non-Urban – green
    1: "#F39C12",   # Semi-Urban – orange
    2: "#E74C3C",   # Urban – red
}
CLASS_NAMES = {0: "Non-Urban", 1: "Semi-Urban", 2: "Urban"}


def create_base_map(center=None, zoom=11, tiles="OpenStreetMap"):
    """
    Create a base Folium map centered on Vijayawada.

    Args:
        center (list | None): [lat, lon] center coordinates.
        zoom (int): Initial zoom level.
        tiles (str): Tile provider.

    Returns:
        folium.Map
    """
    if not FOLIUM_AVAILABLE:
        raise ImportError("folium is required. Install: pip install folium")

    center = center or VIJAYAWADA_CENTER
    m = folium.Map(location=center, zoom_start=zoom, tiles=tiles)

    # Add satellite tile layer
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Google Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    return m


def add_classification_overlay(
    m,
    prediction: np.ndarray,
    bounds: list,
    layer_name="Urban Classification",
    opacity=0.6,
):
    """
    Add a classification map as a coloured image overlay.

    Args:
        m (folium.Map): Base map.
        prediction (np.ndarray): (H, W) integer class map.
        bounds (list): [[south, west], [north, east]] bounding box.
        layer_name (str): Layer label.
        opacity (float): Overlay transparency.

    Returns:
        folium.Map: Map with overlay added.
    """
    if not FOLIUM_AVAILABLE:
        raise ImportError("folium is required.")

    # Convert integer prediction to RGBA image
    h, w = prediction.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cls, hex_color in CLASS_COLORS.items():
        # Parse hex to RGB
        rgb = tuple(int(hex_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        mask = prediction == cls
        rgba[mask, :3] = rgb
        rgba[mask, 3] = int(opacity * 255)

    import io
    import base64
    from PIL import Image

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/png;base64,{encoded}"

    folium.raster_layers.ImageOverlay(
        image=data_url,
        bounds=bounds,
        opacity=opacity,
        name=layer_name,
    ).add_to(m)

    return m


def add_change_heatmap(m, change_map: np.ndarray, bounds: list, layer_name="Urban Growth"):
    """
    Add a change detection heatmap using Folium's HeatMap plugin.

    Args:
        m (folium.Map): Base map.
        change_map (np.ndarray): (H, W) float change intensity map.
        bounds (list): [[south, west], [north, east]].
        layer_name (str): Layer label.

    Returns:
        folium.Map
    """
    if not FOLIUM_AVAILABLE:
        raise ImportError("folium is required.")

    south, west = bounds[0]
    north, east = bounds[1]
    h, w = change_map.shape

    lat_coords = np.linspace(south, north, h)
    lon_coords = np.linspace(west, east, w)

    # Sample only non-zero change pixels for heatmap performance
    data = []
    for i in range(h):
        for j in range(w):
            val = float(change_map[i, j])
            if val > 0.05:
                data.append([lat_coords[i], lon_coords[j], val])

    fg = folium.FeatureGroup(name=layer_name)
    folium_plugins.HeatMap(data, radius=8, blur=10, max_zoom=1).add_to(fg)
    fg.add_to(m)

    return m


def add_legend(m):
    """Add an urban class legend to the map."""
    if not FOLIUM_AVAILABLE:
        return m

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 12px; border-radius: 6px;
                border: 2px solid #ccc; font-family: Arial; font-size: 13px;">
        <b>Urban Classification</b><br>
        <span style="background:#2ECC71; padding:2px 12px; margin-right:4px"></span> Non-Urban<br>
        <span style="background:#F39C12; padding:2px 12px; margin-right:4px"></span> Semi-Urban<br>
        <span style="background:#E74C3C; padding:2px 12px; margin-right:4px"></span> Urban<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def save_map(m, output_path: str):
    """
    Save Folium map to an HTML file.

    Args:
        m (folium.Map): Folium map object.
        output_path (str): Output HTML file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    folium.LayerControl().add_to(m)
    m.save(str(output_path))
    logger.info("Map saved → %s", output_path)


def create_vijayawada_classification_map(
    prediction: np.ndarray,
    bounds: list = None,
    output_path: str = "results/visualizations/vijayawada_classification.html",
    opacity: float = 0.65,
):
    """
    Generate and save a complete Vijayawada urban classification map.

    Args:
        prediction (np.ndarray): (H, W) classification map.
        bounds (list): [[S, W], [N, E]] or None (uses default Vijayawada bounds).
        output_path (str): Output HTML file path.
        opacity (float): Overlay opacity.

    Returns:
        folium.Map
    """
    if bounds is None:
        bounds = [[16.3, 80.4], [16.8, 80.9]]

    m = create_base_map()
    m = add_classification_overlay(m, prediction, bounds, opacity=opacity)
    m = add_legend(m)
    save_map(m, output_path)
    return m


def create_gee_map(year=2023, center=None):
    """
    Create an interactive GEE-backed map for Vijayawada (requires geemap).

    Args:
        year (int): Year for Sentinel-2 imagery.
        center (list | None): [lat, lon].

    Returns:
        geemap.Map | None
    """
    if not GEEMAP_AVAILABLE:
        logger.error("geemap not installed. Install: pip install geemap")
        return None

    try:
        import ee
        ee.Initialize()
    except Exception as exc:
        logger.error("GEE initialization failed: %s", exc)
        return None

    center = center or VIJAYAWADA_CENTER
    m = geemap.Map(center=center, zoom=11)

    roi = ee.Geometry.Point(center[::-1]).buffer(30000)

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .median()
        .clip(roi)
    )

    m.addLayer(
        s2,
        {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "gamma": 1.4},
        f"Sentinel-2 RGB {year}",
    )

    ndbi = s2.normalizedDifference(["B11", "B8"])
    m.addLayer(
        ndbi,
        {"min": -0.5, "max": 0.5, "palette": ["blue", "gray", "red"]},
        "NDBI (Built-up)",
    )

    return m
