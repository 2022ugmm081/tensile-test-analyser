import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from scipy.stats import linregress
from flask import Flask
from dash import Dash, dcc, html, Input, Output

# --- Segment 1: Data Processing ---
def process_csv_data(contents, filename):
    """Process the uploaded CSV file and validate required columns."""
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        required_columns = ['Load_kN', 'Displacement_mm']
        if not all(col in data.columns for col in required_columns):
            return None, "CSV must contain 'Load_kN' and 'Displacement_mm' columns."
        return data, None
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

# --- Segment 2: Mechanical Properties Calculation ---
def calculate_mechanical_properties(data, diameter_mm, gauge_length_mm):
    """Calculate mechanical properties using deviation from linearity."""
    load_kN = data['Load_kN'].values
    displacement_mm = data['Displacement_mm'].values
    load_N = load_kN * 1000
    radius_mm = diameter_mm / 2
    A_0_mm2 = np.pi * radius_mm**2
    stress_MPa = load_N / A_0_mm2
    strain = displacement_mm / gauge_length_mm
    
    # Find deviation from linearity for Young's Modulus and Yield Stress
    linear_points = int(len(stress_MPa) * 0.1)
    slope, intercept, _, _, _ = linregress(strain[:linear_points], stress_MPa[:linear_points])
    predicted_stress = slope * strain + intercept
    deviation = np.abs(stress_MPa - predicted_stress) / (stress_MPa + 1e-6)  # Avoid division by zero
    deviation_threshold = 0.05  # 5% deviation
    yield_index = np.where(deviation > deviation_threshold)[0][0]
    
    # Young's Modulus: stress/strain at deviation point
    youngs_modulus_MPa = stress_MPa[yield_index] / strain[yield_index]
    
    # Yield Stress: stress at deviation point
    yield_stress_MPa = stress_MPa[yield_index]
    
    # Ultimate Tensile Strength
    uts_MPa = np.max(stress_MPa)
    
    # Failure Strain: use max displacement
    failure_strain = (np.max(displacement_mm) / gauge_length_mm) * 100
    
    return {
        'area_mm2': A_0_mm2,
        'youngs_modulus_MPa': youngs_modulus_MPa,
        'yield_stress_MPa': yield_stress_MPa,
        'uts_MPa': uts_MPa,
        'failure_strain_percent': failure_strain,
        'load_displacement_data': {'displacement': displacement_mm.tolist(), 'load': load_kN.tolist()},
        'stress_strain_data': {'strain': strain.tolist(), 'stress': stress_MPa.tolist()}
    }

# --- Segment 3: Plotting ---
def generate_plots(load_displacement_data, stress_strain_data):
    """Generate matplotlib plots with improved styling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='#f0f4f8')
    
    # Load vs. Displacement
    ax1.plot(load_displacement_data['displacement'], load_displacement_data['load'], 
             color='#1f77b4', linewidth=2, label='Load vs. Displacement')
    ax1.set_xlabel('Displacement (mm)', fontsize=12, color='#333')
    ax1.set_ylabel('Load (kN)', fontsize=12, color='#333')
    ax1.set_title('Load vs. Displacement Curve', fontsize=14, color='#333')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    ax1.set_facecolor('#e6f3ff')
    
    # Stress vs. Strain
    ax2.plot(stress_strain_data['strain'], stress_strain_data['stress'], 
             color='#ff7f0e', linewidth=2, label='Stress vs. Strain')
    ax2.set_xlabel('Strain (unitless)', fontsize=12, color='#333')
    ax2.set_ylabel('Stress (MPa)', fontsize=12, color='#333')
    ax2.set_title('Stress vs. Strain Curve', fontsize=14, color='#333')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    ax2.set_facecolor('#fff5e6')
    
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return plot_base64

# --- Segment 4: Flask and Dash App Setup ---
server = Flask(__name__)
app = Dash(__name__, server=server)

# Enhanced Dash layout with colors
app.layout = html.Div(
    style={'backgroundColor': '#f0f4f8', 'padding': '20px', 'fontFamily': 'Arial'},
    children=[
        html.H1("Mechanical Properties Calculator", 
                style={'textAlign': 'center', 'color': '#1f77b4', 'marginBottom': '20px'}),
        html.Div([
            html.Label("Initial Diameter (mm):", style={'color': '#333', 'fontWeight': 'bold'}),
            dcc.Input(id='diameter', type='number', value=6, step=0.1, 
                      style={'margin': '10px', 'width': '200px', 'padding': '5px'}),
        ], style={'backgroundColor': '#e6f3ff', 'padding': '10px', 'borderRadius': '5px'}),
        html.Div([
            html.Label("Initial Gauge Length (mm):", style={'color': '#333', 'fontWeight': 'bold'}),
            dcc.Input(id='gauge_length', type='number', value=32.2, step=0.1, 
                      style={'margin': '10px', 'width': '200px', 'padding': '5px'}),
        ], style={'backgroundColor': '#e6f3ff', 'padding': '10px', 'borderRadius': '5px', 'marginTop': '10px'}),
        html.Div([
            html.Label("Displacement Rate (mm/s):", style={'color': '#333', 'fontWeight': 'bold'}),
            dcc.Input(id='displacement_rate', type='number', value=2, step=0.1, 
                      style={'margin': '10px', 'width': '200px', 'padding': '5px'}),
        ], style={'backgroundColor': '#e6f3ff', 'padding': '10px', 'borderRadius': '5px', 'marginTop': '10px'}),
        html.Div([
            html.Label("Upload CSV File:", style={'color': '#333', 'fontWeight': 'bold'}),
            dcc.Upload(id='upload-data', 
                       children=html.Button('Upload File', style={'backgroundColor': '#1f77b4', 'color': 'white', 'padding': '8px', 'border': 'none', 'borderRadius': '5px'}),
                       style={'margin': '10px'}),
        ], style={'backgroundColor': '#e6f3ff', 'padding': '10px', 'borderRadius': '5px', 'marginTop': '10px'}),
        html.Button('Calculate', id='calculate-button', n_clicks=0, 
                    style={'margin': '20px', 'backgroundColor': '#ff7f0e', 'color': 'white', 'padding': '10px 20px', 'border': 'none', 'borderRadius': '5px'}),
        html.Div(id='output-container', style={'marginTop': '20px', 'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
        dcc.Graph(id='load-displacement-plot', style={'marginTop': '20px'}),
        dcc.Graph(id='stress-strain-plot', style={'marginTop': '20px'}),
    ]
)

@app.callback(
    [Output('output-container', 'children'),
     Output('load-displacement-plot', 'figure'),
     Output('stress-strain-plot', 'figure')],
    [Input('calculate-button', 'n_clicks')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('diameter', 'value'),
     Input('gauge_length', 'value'),
     Input('displacement_rate', 'value')]
)
def update_output(n_clicks, contents, filename, diameter, gauge_length, displacement_rate):
    """Handle user inputs, process data, calculate properties, and update plots."""
    if n_clicks == 0 or contents is None:
        return html.P("Please upload a file and click Calculate.", style={'color': '#d32f2f'}), {}, {}
    if not all(isinstance(x, (int, float)) and x > 0 for x in [diameter, gauge_length, displacement_rate]):
        return html.P("Inputs must be positive numbers.", style={'color': '#d32f2f'}), {}, {}
    
    data, error = process_csv_data(contents, filename)
    if error:
        return html.P(error, style={'color': '#d32f2f'}), {}, {}
    
    results = calculate_mechanical_properties(data, diameter, gauge_length)
    plot_base64 = generate_plots(results['load_displacement_data'], results['stress_strain_data'])
    
    output = html.Div([
        html.H3("Results:", style={'color': '#1f77b4'}),
        html.P(f"Initial Cross-Sectional Area: {results['area_mm2']:.3f} mm²", style={'color': '#333'}),
        html.P(f"Young's Modulus: {results['youngs_modulus_MPa']:.2f} MPa", style={'color': '#333'}),
        html.P(f"Yield Stress: {results['yield_stress_MPa']:.2f} MPa", style={'color': '#333'}),
        html.P(f"Ultimate Tensile Strength: {results['uts_MPa']:.2f} MPa", style={'color': '#333'}),
        html.P(f"Failure Strain: {results['failure_strain_percent']:.2f}%", style={'color': '#333'}),
        html.Img(src=f"data:image/png;base64,{plot_base64}", style={'width': '100%', 'marginTop': '20px'})
    ])
    
    load_displacement_fig = {
        'data': [{
            'x': results['load_displacement_data']['displacement'],
            'y': results['load_displacement_data']['load'],
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Load vs. Displacement',
            'line': {'color': '#1f77b4'}
        }],
        'layout': {
            'title': {'text': 'Load vs. Displacement Curve', 'x': 0.5, 'xanchor': 'center'},
            'xaxis': {'title': 'Displacement (mm)', 'gridcolor': '#e0e0e0'},
            'yaxis': {'title': 'Load (kN)', 'gridcolor': '#e0e0e0'},
            'showlegend': True,
            'plot_bgcolor': '#e6f3ff',
            'paper_bgcolor': '#f0f4f8'
        }
    }
    
    stress_strain_fig = {
        'data': [{
            'x': results['stress_strain_data']['strain'],
            'y': results['stress_strain_data']['stress'],
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Stress vs. Strain',
            'line': {'color': '#ff7f0e'}
        }],
        'layout': {
            'title': {'text': 'Stress vs. Strain Curve', 'x': 0.5, 'xanchor': 'center'},
            'xaxis': {'title': 'Strain (unitless)', 'gridcolor': '#e0e0e0'},
            'yaxis': {'title': 'Stress (MPa)', 'gridcolor': '#e0e0e0'},
            'showlegend': True,
            'plot_bgcolor': '#fff5e6',
            'paper_bgcolor': '#f0f4f8'
        }
    }
    
    return output, load_displacement_fig, stress_strain_fig

@server.route('/')
def index():
    return app.index()

if __name__ == '__main__':
    server.run(debug=True)