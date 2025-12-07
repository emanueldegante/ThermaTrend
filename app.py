# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 23:06:09 2025

@author: emanu
"""

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import base64
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import plotly.io as pio

# ------------------- Email Configuration -------------------
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
EMAIL_SENDER = "geophysicsunb@gmail.com"
EMAIL_PASSWORD = "qmyzqduwojfbwjen"  # Use Gmail App Password

# -------------------- Load Data --------------------
df = pd.read_csv("DTS_daily_Total.csv", index_col=0, parse_dates=True)
df.index = pd.to_datetime(df.index)
r = np.array(df.columns[1:], dtype=np.float64)  # Elevation reference

# -------------------- Helper Functions --------------------
def extract_date_across_years(df, month, day):
    mask = (df.index.month == month) & (df.index.day == day)
    return df.loc[mask]

def df_plot_demeaned(same_day):
    time = pd.to_datetime(same_day.index).date
    df_t = np.transpose(same_day)
    df_t2 = np.array(df_t)
    df_new = pd.DataFrame(df_t2, columns=time)
    df_mean = np.nanmean(df_t2, axis=1).reshape(-1,1)
    df_demeaned = df_t2 - df_mean
    df_dem = pd.DataFrame(df_demeaned, columns=time)
    return df_new, df_dem, df_mean

# -------------------- Dash App --------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
all_years = sorted(df.index.year.unique())

# -------------------- Layout --------------------
app.layout = dbc.Container([
    html.H2("ThermaTrace", className="text-center text-white my-3"),

    html.Label("Select Date:  ", style={'color': 'white'}),
    dcc.DatePickerSingle(
        id='date-picker',
        min_date_allowed=df.index.min(),
        max_date_allowed=df.index.max(),
        date=df.index.min()
    ),

    html.Br(),
    html.Label("Select Years to Show:  ", style={'color': 'white'}),
    dcc.Dropdown(
        id='year-dropdown',
        options=[{'label': str(year), 'value': year} for year in all_years],
        multi=True,
        value=all_years
    ),

    html.Br(),
    html.Label("Select Elevation for Time Series:  ", style={'color': 'white'}),
    dcc.Input(
        id='elevation-input',
        type='number',
        value=float(r[0]),
        step=0.1
    ),

    html.Br(),
    html.Label("Select Display Mode:", style={'color': 'white'}),
    dcc.RadioItems(
        id='display-mode',
        options=[{'label': 'DTS Fibre Optic', 'value': 1},
                 {'label': 'Demeaned', 'value': 2}],
        value=1,
        labelStyle={'display': 'inline-block', 'margin-right': '20px', 'color': 'white'}
    ),

    html.Br(),
    dcc.Graph(id='dts-plot', style={'height': '700px'}),
    html.Hr(style={'borderColor': 'white'}),
    html.H3("Temperature Time Series on Selected Date Across Years", style={'color': 'white'}),
    dcc.Graph(id='elevation-time-series', style={'height': '500px'}),

    # Email section
    html.Hr(style={'borderColor': 'white'}),
    html.H3("Send Plots via Email", style={'color': 'white'}),
    dcc.Input(id='recipient-email', type='email', placeholder='Recipient Email', style={'width':'100%', 'marginBottom':'10px'}),
    dbc.Button("Send Email", id='send-email-btn', color='primary', n_clicks=0),
    html.Div(id='email-status', style={'color':'white', 'marginTop':'10px'})
],
style={'backgroundColor': 'orange', 'padding': '20px', 'minHeight': '100vh'})

# -------------------- DTS Plot Callback --------------------
@app.callback(
    Output('dts-plot', 'figure'),
    Input('date-picker', 'date'),
    Input('year-dropdown', 'value'),
    Input('elevation-input', 'value'),
    Input('display-mode', 'value')
)
def update_plot(selected_date, selected_years, selected_elevation, display_mode):
    if selected_date is None or not selected_years:
        return go.Figure()

    selected_date_dt = pd.to_datetime(selected_date)
    month, day = selected_date_dt.month, selected_date_dt.day
    same_day = extract_date_across_years(df, month, day)

    if same_day.empty:
        return go.Figure()

    df_new, df_dem, df_mean = df_plot_demeaned(same_day)
    df_plot = df_new if display_mode == 1 else df_dem
    df_plot.columns = pd.to_datetime(same_day.index).year
    df_plot = df_plot[[year for year in selected_years if year in df_plot.columns]]

    if df_plot.empty:
        return go.Figure()

    fig = go.Figure()

    # ---- RAINBOW COLORS ----
    base_colors = px.colors.sequential.Rainbow
    # interpolate to number of columns:
    rainbow_colors = base_colors * (len(df_plot.columns) // len(base_colors) + 1)
    rainbow_colors = rainbow_colors[:len(df_plot.columns)]
    # -------------------------
    
    selected_year = selected_date_dt.year
    
    for i, col in enumerate(df_plot.columns):
        y_data = r
        x_data = df_plot[col].rolling(window=3, min_periods=1).mean()
    
        # If this trace is for the selected year → make it black
        line_color = "black" if col == selected_year else rainbow_colors[i]
    
        fig.add_trace(go.Scatter(
            y=y_data,
            x=x_data,
            mode='lines',    #'lines+markers
            name=str(col),
            line=dict(color=line_color, width=3 if col == selected_year else 2.5)
        ))

    if selected_elevation is not None:
        idx = np.argmin(np.abs(r - selected_elevation))
        selected_elevation_actual = r[idx]
        fig.add_hline(
            y=selected_elevation_actual,
            line=dict(color='red', width=2, dash='dash'),
            annotation_text=f"Selected Elevation: {selected_elevation_actual:.2f} m",
            annotation_position="top left",
            annotation_font_color="black"
        )

    title_mode = 'Demeaned' if display_mode == 2 else 'Original'

    fig.update_layout(
        title=(
            f"Mactaquac Borehole DTS\nElevation vs Temperature ({title_mode})\n"
            f"Historical for {selected_date_dt.strftime('%B %d')}"
        ),
        yaxis_title="Elevation (m)",
        xaxis_title="Temperature (°C)",
        legend_title="Year",
        template="plotly_white"
    )

    return fig

# -------------------- Elevation Time Series Callback --------------------
@app.callback(
    Output('elevation-time-series', 'figure'),
    Input('elevation-input', 'value'),
    Input('year-dropdown', 'value'),
    Input('date-picker', 'date')
)
def elevation_time_series(selected_elevation, selected_years, selected_date):
    if selected_elevation is None or not selected_years or selected_date is None:
        return go.Figure()
    selected_date_dt = pd.to_datetime(selected_date)
    month, day = selected_date_dt.month, selected_date_dt.day
    same_day = extract_date_across_years(df, month, day)
    idx = np.argmin(np.abs(r - selected_elevation))
    selected_elevation_actual = r[idx]
    years_sorted = sorted(selected_years)
    temps = []
    for year in years_sorted:
        year_data = same_day[same_day.index.year == year]
        temps.append(year_data.iloc[0, idx] if not year_data.empty else np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years_sorted, y=temps,
        mode='lines+markers',
        name=f'Elevation {selected_elevation_actual:.2f} m',
        marker=dict(size=10),
        line=dict(width=2),
        connectgaps=True
    ))
    fig.update_layout(
        title=f"Temperature on {selected_date_dt.strftime('%B %d')} at {selected_elevation_actual:.2f} m",
        xaxis_title="Year",
        yaxis_title="Temperature (°C)",
        template="plotly_white"
    )
    return fig

# -------------------- Email Callback --------------------
@app.callback(
    Output('email-status', 'children'),
    Input('send-email-btn', 'n_clicks'),
    State('recipient-email', 'value'),
    State('dts-plot', 'figure'),
    State('elevation-time-series', 'figure')
)
def send_email(n_clicks, recipient, fig1, fig2):
    if n_clicks == 0:
        return ""
    if not recipient:
        return "Please enter a recipient email."

    try:
        # Convert figures to PNG bytes
        fig1_bytes = pio.to_image(go.Figure(fig1), format='png', width=700, height=500)
        fig2_bytes = pio.to_image(go.Figure(fig2), format='png', width=700, height=500)
        fig1_b64 = base64.b64encode(fig1_bytes).decode()
        fig2_b64 = base64.b64encode(fig2_bytes).decode()

        html_body = f"""
        <html>
            <body>
                <p>AAR Team,<br><br>Here are the DTS plots:</p>
                <b>DTS Temperature vs Depth Plot:</b><br>
                <img src="data:image/png;base64,{fig1_b64}" width="700"><br><br>
                <b>Elevation Time Series:</b><br>
                <img src="data:image/png;base64,{fig2_b64}" width="700"><br><br>
                Regards,<br>DTScope Analytics
            </body>
        </html>
        """

        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = recipient
        msg['Subject'] = "Mactaquac DTS Plots"
        msg.attach(MIMEText(html_body, 'html'))

        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        return "✅ Email sent successfully with inline images!"

    except Exception as e:
        return f"❌ Error sending email: {str(e)}"

# -------------------- Run App --------------------
if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
