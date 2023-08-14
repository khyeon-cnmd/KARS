from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.transform import dodge
import pandas as pd

# 데이터 생성
data = {'Category': ['A', 'B', 'C', 'D'],
        'Value': [10, 15, 7, 12]}

df = pd.DataFrame(data)

# ColumnDataSource 생성
source = ColumnDataSource(df)

# 도화지 생성
p = figure(x_range=df['Category'], plot_height=250, title='Horizontal Bar Chart')

# 가로 막대 그래프 그리기
p.hbar(y='Category', right='Value', source=source, height=0.5, color='navy')

# 그래프 보여주기
show(p)
