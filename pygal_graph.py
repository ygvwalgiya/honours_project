import pygal
from pygal.style import BlueStyle

xy_chart = pygal.XY(stroke=False, style=BlueStyle)
xy_chart.title = 'XY Cosinus'
xy_chart.add('Current vs Weight',[(335.22,3.3),(416.75,4.1),(497.86,5),(579.18,5.7),(660.3,6.6),(741.95,7.6),(823.14,8.3),(904.57,9.3),(985.22,9.9)])

xy_chart.render_to_file('website_embed.svg')