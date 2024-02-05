import plotly.express as px
import pandas as pd

# Sample data: a task with discrete durations
tasks = [
    {"Task": "Task A", "Start": "2021-01-01 10:00:00", "Finish": "2021-01-01 11:00:00"},
    {"Task": "Task A", "Start": "2021-01-01 12:00:00", "Finish": "2021-01-01 13:00:00"},
    {"Task": "Task A", "Start": "2021-01-01 14:00:00", "Finish": "2021-01-01 15:00:00"},
    {"Task": "Task B", "Start": "2021-01-01 10:30:00", "Finish": "2021-01-01 11:30:00"},
    {"Task": "Task B", "Start": "2021-01-01 12:30:00", "Finish": "2021-01-01 13:30:00"},
]

# Convert the list of tasks to a DataFrame
df = pd.DataFrame(tasks)

# Convert the start and end times to datetime
df['Start'] = pd.to_datetime(df['Start'])
df['Finish'] = pd.to_datetime(df['Finish'])

# Create a Gantt chart
fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", labels={"Task": "Task Name"})
fig.update_yaxes(categoryorder="total ascending")

# Show the figure
fig.show()
