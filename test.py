import networkx as nx

from bokeh.palettes import Category20_20
from bokeh.plotting import figure, from_networkx, show

G = nx.random_geometric_graph(100, 1.5)

p = figure(x_range=(-2, 2), y_range=(-2, 2),
           x_axis_location=None, y_axis_location=None,
           tools="hover", tooltips="index: @index")
p.grid.grid_line_color = None

graph = from_networkx(G, nx.spring_layout, scale=1.8, center=(0,0))
p.renderers.append(graph)

# Add some new columns to the node renderer data source
graph.node_renderer.data_source.data['index'] = list(range(len(G)))
graph.node_renderer.data_source.data['colors'] = Category20_20

graph.node_renderer.glyph.update(size=20, fill_color="colors")
show(p)
