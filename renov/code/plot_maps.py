import folium
from folium.plugins import StripePattern
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pyproj
import branca.colormap as cm


def single_map(
    data,
    col,
    measure,
    cols_hover,
    col_name=None,
    cols_hover_name=None,
    loc=[46.883, 8.52],
    height="100%",
    width="100%",
    vmax=2000,
    text_div=None,
    caption=None,
    map=None,
    name_map=None,
):
    if col_name is None:
        col_name = col
    if cols_hover_name is None:
        cols_hover_name = cols_hover
    if name_map is None:
        name_map = ""
    # We create another map called map
    if map is None:
        map = folium.Map(
            location=loc,
            zoom_start=8,
            scrollWheelZoom=False,
            tiles="cartodbpositron",
            height=height,
            width=width,
        )

    data["geometry"] = data.geometry.simplify(tolerance=200, preserve_topology=True)
    data[cols_hover] = np.round(data[cols_hover], 2)
    colormap = cm.linear.OrRd_09.scale(vmin=0.0, vmax=vmax).to_step(n=8)

    colormap.caption = f"{col_name} (measure {measure})" if caption is None else caption

    def get_color(value):
        if value is None:
            return "#ffffff"  # MISSING -> gray
        else:
            return colormap(value)

    style_function = lambda x: {
        "weight": 0.5,
        "color": "black",
        "fillColor": get_color(x["properties"][col]),
        "fillOpacity": 0.8,
        "line_opacity": 0.9,
    }
    NIL = folium.features.GeoJson(
        data,
        style_function=style_function,
        control=False,
        show=True,
        name=name_map,
    )
    colormap.add_to(map)
    map.add_child(NIL)

    # Add hover functionality.
    style_function = lambda x: {
        "fillColor": "#ffffff",
        "color": "#000000",
        "fillOpacity": 0.1,
        "weight": 0.2,
    }
    highlight_function = lambda x: {
        "fillColor": "#000000",
        "color": "#000000",
        "fillOpacity": 0.70,
        "weight": 1.0,
    }
    fields = ["NAME"]
    if not isinstance(cols_hover, list):
        cols_hover = [cols_hover]
    fields.extend(cols_hover)
    alias = ["Name"]
    if not isinstance(cols_hover_name, list):
        cols_hover_name = [cols_hover_name]
    alias.extend(cols_hover_name)

    NIL = folium.features.GeoJson(
        data=data,
        style_function=style_function,
        control=False,
        highlight_function=highlight_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=fields,
            aliases=alias,
            style=(
                "background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"
            ),
        ),
    )
    map.add_child(NIL)
    map.keep_in_front(NIL)

    # Here we add cross-hatching (crossing lines) to display the Null values.
    gdf_nans = data[data[col].isnull()]
    if len(gdf_nans) > 0:
        sp = StripePattern(angle=45, color="grey", space_color="white")
        sp.add_to(map)
        missing = folium.features.GeoJson(
            name=f"{name_map} NaN values",
            data=gdf_nans,
            style_function=lambda x: {
                "fillPattern": sp,
                "color": "black",
                "weight": 2,
                "opacity": 0.7,
            },
            show=True,
            control=True,
        )
        map.add_child(missing)

    # Add dark and
    folium.TileLayer("cartodbdark_matter", name="dark mode", control=True).add_to(map)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(map)

    # We add a layer controller.
    folium.LayerControl(collapsed=True).add_to(map)
    if text_div is not None:
        map.get_root().html.add_child(
            folium.Element(
                f"""
        <div style="position: fixed;
                left: 50%;
                bottom: 20px;
                transform: translate(-50%, -50%);
                margin: 0 auto; 
                padding: 10px;
            background-color:white; border:2px solid grey;z-index: 900;">
            <h5>{text_div}</h5>
        </div>
        """
            )
        )

    return map


def map_with_progress_bar(
    data, col, cols_hover_format, loc=[46.883, 8.52], height=500, width=750, vmax=2000
):
    data["geometry"] = data.geometry.simplify(tolerance=200, preserve_topology=True)
    data.to_crs(pyproj.CRS.from_epsg(4326), inplace=True)
    data_ = data[~data.isna().any(axis=1)].copy().sort_values("Year")
    if vmax is None:
        vmax = data_[col].max()
    geojson = data_.__geo_interface__
    fig = px.choropleth_mapbox(
        data_,
        geojson=geojson,
        locations="Canton",
        color=col,
        color_continuous_scale="Reds",
        featureidkey="properties.Canton",
        mapbox_style="carto-positron",
        opacity=0.8,
        center={"lat": loc[0], "lon": loc[1]},
        zoom=6,
        animation_frame="Year",
        hover_data=cols_hover_format,
        range_color=[0, vmax],
        height=height,
        width=width,
    )
    fig.update_geos(
        fitbounds="locations",
        visible=False,
    )
    fig.update_layout(
        margin={"r": 100, "t": 0, "l": 0, "b": 0},
        hovermode="x",
        coloraxis2={"colorscale": [[0, "gray"], [1, "gray"]], "showscale": False},
    )
    for i, fr in enumerate(fig.frames):
        data_ = data.loc[
            data[col].isna() & (data["Year"] == 2017 + i),
            ["Canton", "Year", "geometry"],
        ].copy()
        geojson = data_.__geo_interface__
        tr_missing = (
            px.choropleth_mapbox(
                data_.assign(color=1),
                geojson=geojson,
                color="color",
                locations="Canton",
                featureidkey="properties.Canton",
                color_continuous_scale=[[0, "gray"], [1, "gray"]],
            )
            .update_traces(hovertemplate="missing: %{location}", coloraxis="coloraxis2")
            .data[0]
        )
        fr.update(data=[fr.data[0], tr_missing])
    # Re-construct the figure...
    fig = go.Figure(data=fig.frames[0].data, layout=fig.layout, frames=fig.frames)
    return fig
