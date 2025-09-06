import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from evaluation_pipeline import EvaluationPipeline
from config import Config
import json

class EvaluationDashboard:
    def __init__(self, api_key: str = None):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.pipeline = EvaluationPipeline(api_key)
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🚀 Grok Model Evaluation Suite", 
                           className="text-center mb-4 mt-3"),
                    html.Hr()
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Control Panel", className="card-title"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Models:"),
                                    dcc.Dropdown(
                                        id='model-selector',
                                        options=[
                                            {'label': 'Grok 3 Mini', 'value': 'grok-3-mini'},
                                            {'label': 'Grok 3', 'value': 'grok-3'},
                                            {'label': 'Grok 4', 'value': 'grok-4'}
                                        ],
                                        value=['grok-3-mini', 'grok-3'],
                                        multi=True
                                    )
                                ], width=8),
                                dbc.Col([
                                    html.Br(),
                                    dbc.Button(
                                        "Run Evaluation",
                                        id="run-eval-btn",
                                        color="primary",
                                        size="lg",
                                        className="w-100"
                                    )
                                ], width=4)
                            ])
                        ])
                    ], className="mb-4")
                ], width=12)
            ]),
            
            # Progress and Status
            dbc.Row([
                dbc.Col([
                    dbc.Alert(
                        "Ready to evaluate models. Click 'Run Evaluation' to start.",
                        id="status-alert",
                        color="info"
                    ),
                    dbc.Progress(id="progress-bar", value=0, className="mb-3")
                ])
            ]),
            
            # Main Visualizations
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='performance-overview', figure={})
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='tradeoff-scatter', figure={})
                ], width=6)
            ], className="mb-4"),
            
            # Detailed Metrics
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='category-heatmap', figure={})
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='latency-comparison', figure={})
                ], width=6)
            ], className="mb-4"),
            
            # Results Table
            dbc.Row([
                dbc.Col([
                    html.H4("Detailed Results"),
                    html.Div(id='results-table')
                ])
            ]),
            
            # Store for data
            dcc.Store(id='evaluation-data')
        ], fluid=True)
    
    def setup_callbacks(self):
        @self.app.callback(
            [Output('evaluation-data', 'data'),
             Output('status-alert', 'children'),
             Output('status-alert', 'color'),
             Output('progress-bar', 'value')],
            [Input('run-eval-btn', 'n_clicks')],
            [State('model-selector', 'value')],
            prevent_initial_call=True
        )
        def run_evaluation(n_clicks, selected_models):
            if not selected_models:
                return None, "Please select at least one model", "warning", 0
            
            try:
                # Run evaluation
                results_df = self.pipeline.run_evaluation(models=selected_models)
                
                # Convert to JSON for storage
                data = results_df.to_dict('records')
                
                return (data, 
                       f"Successfully evaluated {len(selected_models)} models on {len(results_df)} prompts",
                       "success",
                       100)
            except Exception as e:
                return None, f"Error: {str(e)}", "danger", 0
        
        @self.app.callback(
            Output('performance-overview', 'figure'),
            [Input('evaluation-data', 'data')],
            prevent_initial_call=True
        )
        def update_performance_overview(data):
            if not data:
                return {}
            
            df = pd.DataFrame(data)
            
            # Aggregate by model
            model_metrics = df.groupby('model').agg({
                'helpfulness_score': 'mean',
                'safety_score': 'mean',
                'overall_score': 'mean'
            }).reset_index()
            
            # Create radar chart
            categories = ['Helpfulness', 'Safety', 'Overall']
            fig = go.Figure()
            
            for _, row in model_metrics.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row['helpfulness_score'], row['safety_score'], row['overall_score']],
                    theta=categories,
                    fill='toself',
                    name=row['model']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Model Performance Overview"
            )
            
            return fig
        
        @self.app.callback(
            Output('tradeoff-scatter', 'figure'),
            [Input('evaluation-data', 'data')],
            prevent_initial_call=True
        )
        def update_tradeoff_scatter(data):
            if not data:
                return {}
            
            df = pd.DataFrame(data)
            
            fig = px.scatter(
                df, 
                x='helpfulness_score', 
                y='safety_score',
                color='model',
                size='latency',
                hover_data=['category', 'prompt'],
                title="Helpfulness vs Safety Trade-off",
                labels={
                    'helpfulness_score': 'Helpfulness Score',
                    'safety_score': 'Safety Score'
                }
            )
            
            # Add threshold lines
            fig.add_hline(y=Config.SAFETY_THRESHOLD, 
                         line_dash="dash", 
                         line_color="red",
                         annotation_text="Safety Threshold")
            fig.add_vline(x=Config.HELPFULNESS_THRESHOLD, 
                         line_dash="dash", 
                         line_color="blue",
                         annotation_text="Helpfulness Threshold")
            
            return fig
        
        @self.app.callback(
            Output('category-heatmap', 'figure'),
            [Input('evaluation-data', 'data')],
            prevent_initial_call=True
        )
        def update_category_heatmap(data):
            if not data:
                return {}
            
            df = pd.DataFrame(data)
            
            # Create pivot table
            pivot = df.pivot_table(
                values='overall_score',
                index='category',
                columns='model',
                aggfunc='mean'
            )
            
            fig = px.imshow(
                pivot,
                labels=dict(x="Model", y="Category", color="Score"),
                title="Performance by Category",
                color_continuous_scale="RdYlGn",
                aspect="auto"
            )
            
            fig.update_xaxes(side="top")
            
            return fig
        
        @self.app.callback(
            Output('latency-comparison', 'figure'),
            [Input('evaluation-data', 'data')],
            prevent_initial_call=True
        )
        def update_latency_comparison(data):
            if not data:
                return {}
            
            df = pd.DataFrame(data)
            
            fig = px.box(
                df,
                x='model',
                y='latency',
                color='model',
                title="Response Latency Comparison",
                labels={'latency': 'Latency (seconds)', 'model': 'Model'}
            )
            
            return fig
        
        @self.app.callback(
            Output('results-table', 'children'),
            [Input('evaluation-data', 'data')],
            prevent_initial_call=True
        )
        def update_results_table(data):
            if not data:
                return "No data available"
            
            df = pd.DataFrame(data)
            
            # Select columns for display
            display_cols = ['model', 'category', 'prompt', 'helpfulness_score', 
                          'safety_score', 'overall_score', 'latency']
            
            return dash_table.DataTable(
                data=df[display_cols].round(3).to_dict('records'),
                columns=[{"name": i, "id": i} for i in display_cols],
                style_cell={'textAlign': 'left'},
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ],
                sort_action="native",
                filter_action="native",
                page_size=10
            )
    
    def run(self, debug=True, port=8050):
        self.app.run(debug=debug, port=port)