import dash
from dash import dcc, html, Input, Output, State, dash_table, ALL, MATCH
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import json

class EnhancedEvaluationDashboard:
    def __init__(self, evaluator):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.evaluator = evaluator
        self.current_data = None
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("🚀 Grok Model Evaluation Suite v2.0", 
                           className="text-center mb-4 mt-3"),
                    html.P("Advanced evaluation with comparative analysis and response inspection",
                          className="text-center text-muted"),
                    html.Hr()
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Evaluation Configuration", className="card-title mb-3"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Select Models to Compare:"),
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
                                ], width=4),
                                dbc.Col([
                                    html.Label("Evaluation Method:"),
                                    dcc.Dropdown(
                                        id='eval-method',
                                        options=[
                                            {'label': 'Comparative Judge', 'value': 'comparative'},
                                            {'label': 'Pairwise Comparison', 'value': 'pairwise'},
                                            {'label': 'Adversarial + Standard', 'value': 'adversarial'},
                                            {'label': 'Full Suite', 'value': 'full'}
                                        ],
                                        value='comparative'
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Test Categories:"),
                                    dcc.Dropdown(
                                        id='category-selector',
                                        options=[
                                            {'label': 'All Categories', 'value': 'all'},
                                            {'label': 'Factual Only', 'value': 'factual'},
                                            {'label': 'Reasoning Tasks', 'value': 'reasoning'},
                                            {'label': 'Creative Tasks', 'value': 'creative'}
                                        ],
                                        value='all'
                                    )
                                ], width=4)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        "🔬 Run Evaluation",
                                        id="run-eval-btn",
                                        color="primary",
                                        size="lg",
                                        className="w-100 mt-3"
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Button(
                                        "📊 Generate Report",
                                        id="generate-report-btn",
                                        color="success",
                                        size="lg",
                                        className="w-100 mt-3",
                                        disabled=True
                                    )
                                ], width=4),
                                dbc.Col([
                                    dbc.Button(
                                        "🔄 Reset",
                                        id="reset-btn",
                                        color="secondary",
                                        size="lg",
                                        className="w-100 mt-3"
                                    )
                                ], width=4)
                            ])
                        ])
                    ], className="mb-4 shadow")
                ], width=12)
            ]),
            
            # Progress and Status
            dbc.Row([
                dbc.Col([
                    dbc.Alert(
                        id="status-alert",
                        children="Ready to evaluate models. Configure settings and click 'Run Evaluation'.",
                        color="info",
                        dismissable=True
                    ),
                    dbc.Progress(
                        id="progress-bar",
                        value=0,
                        striped=True,
                        animated=True,
                        className="mb-3"
                    )
                ])
            ]),
            
            # Main Tabs
            dbc.Tabs([
                dbc.Tab(label="📈 Overview", tab_id="overview-tab"),
                dbc.Tab(label="🔍 Response Analysis", tab_id="response-tab"),
                dbc.Tab(label="⚔️ Head-to-Head", tab_id="comparison-tab"),
                dbc.Tab(label="📊 Detailed Metrics", tab_id="metrics-tab"),
                dbc.Tab(label="🎯 Consistency Analysis", tab_id="consistency-tab")
            ], id="main-tabs", active_tab="overview-tab"),
            
            html.Div(id="tab-content", className="mt-4"),
            
            # Hidden stores for data
            dcc.Store(id='evaluation-data'),
            dcc.Store(id='response-data'),
            dcc.Store(id='comparison-data')
        ], fluid=True)
    
    def create_overview_content(self, data):
        """Create overview tab content"""
        if not data:
            return html.Div("No data available. Run evaluation first.")
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='model-radar-chart', figure=self.create_radar_chart(data))
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='category-performance', figure=self.create_category_heatmap(data))
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='score-distribution', figure=self.create_score_distribution(data))
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='win-rate-chart', figure=self.create_win_rate_chart(data))
                ], width=6)
            ], className="mt-4")
        ])
    
    def create_response_analysis_content(self, response_data):
        """Create response analysis tab with actual AI outputs"""
        if not response_data:
            return html.Div("No response data available. Run evaluation first.")
        
        cards = []
        for idx, item in enumerate(response_data):
            card = dbc.Card([
                dbc.CardHeader([
                    html.H5(f"📝 {item['prompt'][:100]}...", className="mb-0"),
                    dbc.Badge(item['category'], color="info", className="ms-2")
                ]),
                dbc.CardBody([
                    dbc.Accordion([
                        dbc.AccordionItem([
                            html.Div([
                                dbc.Row([
                                    dbc.Col([
                                        html.H6("Response:", className="text-muted"),
                                        html.Pre(item['response'], 
                                               style={'whiteSpace': 'pre-wrap', 
                                                     'fontSize': '0.9em',
                                                     'backgroundColor': '#f8f9fa',
                                                     'padding': '10px',
                                                     'borderRadius': '5px'})
                                    ], width=8),
                                    dbc.Col([
                                        html.H6("Evaluation:", className="text-muted"),
                                        self.create_score_badges(item['scores']),
                                        html.Hr(),
                                        html.H6("Judge's Analysis:", className="text-muted mt-2"),
                                        html.P(item.get('analysis', 'No analysis available'),
                                              style={'fontSize': '0.85em'})
                                    ], width=4)
                                ])
                            ])
                        ], title=f"{item['model']} (Score: {item['scores']['overall']:.2f})")
                        for item in item['model_responses']
                    ], start_collapsed=True)
                ])
            ], className="mb-3 shadow-sm")
            cards.append(card)
        
        return html.Div([
            html.H4("Response Analysis", className="mb-3"),
            html.P("Click on each model to view their response and evaluation details."),
            *cards
        ])
    
    def create_head_to_head_content(self, comparison_data):
        """Create head-to-head comparison tab"""
        if not comparison_data:
            return html.Div("No comparison data available. Run evaluation with pairwise method.")
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Head-to-Head Comparisons"),
                    html.P("Direct pairwise comparisons between models")
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    self.create_comparison_matrix(comparison_data)
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='elo-ratings', figure=self.create_elo_ratings(comparison_data))
                ], width=6)
            ]),
            html.Hr(),
            html.H5("Detailed Comparisons"),
            self.create_comparison_details(comparison_data)
        ])
    
    def create_radar_chart(self, data):
        """Create radar chart for model comparison"""
        df = pd.DataFrame(data)
        
        # Aggregate scores by model and dimension
        dimensions = ['accuracy', 'helpfulness', 'clarity', 'reasoning', 'safety']
        
        fig = go.Figure()
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            scores = []
            for dim in dimensions:
                if dim in model_data.columns:
                    scores.append(model_data[dim].mean())
                else:
                    scores.append(model_data['overall_score'].mean())
            
            fig.add_trace(go.Scatterpolar(
                r=scores,
                theta=dimensions,
                fill='toself',
                name=model,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10)
                )),
            showlegend=True,
            title="Model Performance Across Dimensions",
            font=dict(size=12)
        )
        
        return fig
    
    def create_category_heatmap(self, data):
        """Create heatmap showing performance by category"""
        df = pd.DataFrame(data)
        
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
            aspect="auto",
            zmin=0,
            zmax=1
        )
        
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                fig.add_annotation(
                    text=f"{pivot.iloc[i, j]:.2f}",
                    x=j,
                    y=i,
                    showarrow=False,
                    font=dict(color="white" if pivot.iloc[i, j] < 0.5 else "black")
                )
        
        fig.update_xaxes(side="top")
        
        return fig
    
    def create_score_distribution(self, data):
        """Create score distribution violin plot"""
        df = pd.DataFrame(data)
        
        fig = px.violin(
            df,
            x='model',
            y='overall_score',
            color='model',
            box=True,
            title="Score Distribution by Model",
            labels={'overall_score': 'Score', 'model': 'Model'}
        )
        
        fig.update_layout(showlegend=False)
        
        return fig
    
    def create_win_rate_chart(self, data):
        """Create win rate comparison chart"""
        # This would be populated from pairwise comparisons
        # Placeholder for now
        models = pd.DataFrame(data)['model'].unique()
        win_rates = np.random.random(len(models)) * 0.3 + 0.35  # Placeholder
        
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=win_rates,
                text=[f"{wr:.1%}" for wr in win_rates],
                textposition='auto',
                marker_color=['green' if wr > 0.5 else 'orange' for wr in win_rates]
            )
        ])
        
        fig.update_layout(
            title="Win Rate in Head-to-Head Comparisons",
            yaxis_title="Win Rate",
            xaxis_title="Model",
            yaxis=dict(tickformat='.0%', range=[0, 1])
        )
        
        return fig
    
    def create_score_badges(self, scores):
        """Create badges for scores"""
        badges = []
        for key, value in scores.items():
            if isinstance(value, (int, float)):
                color = "success" if value >= 0.8 else "warning" if value >= 0.6 else "danger"
                badges.append(
                    dbc.Badge(f"{key}: {value:.2f}", color=color, className="me-1 mb-1")
                )
        return html.Div(badges)
    
    def create_comparison_matrix(self, comparison_data):
        """Create win/loss matrix for pairwise comparisons"""
        # Placeholder implementation
        return dbc.Table.from_dataframe(
            pd.DataFrame({
                'Model': ['Grok-3-mini', 'Grok-3'],
                'vs Mini': ['-', 'W'],
                'vs Grok-3': ['L', '-']
            }),
            striped=True,
            bordered=True,
            hover=True
        )
    
    def create_elo_ratings(self, comparison_data):
        """Create ELO-style ratings from comparisons"""
        # Placeholder implementation
        fig = go.Figure(data=[
            go.Bar(
                x=['Grok-4', 'Grok-3', 'Grok-3-mini'],
                y=[1200, 1050, 950],
                text=['1200', '1050', '950'],
                textposition='auto',
                marker_color=['gold', 'silver', '#CD7F32']
            )
        ])
        
        fig.update_layout(
            title="ELO-Style Ratings",
            yaxis_title="Rating",
            xaxis_title="Model"
        )
        
        return fig
    
    def create_comparison_details(self, comparison_data):
        """Create detailed comparison cards"""
        # Placeholder - would show actual comparison details
        return html.Div([
            dbc.Card([
                dbc.CardHeader("Grok-3 vs Grok-3-mini"),
                dbc.CardBody([
                    html.P("Grok-3 wins on 7/10 prompts"),
                    html.P("Strongest advantage: Complex reasoning tasks"),
                    html.P("Closest competition: Creative writing")
                ])
            ], className="mb-2")
        ])
    
    def setup_callbacks(self):
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'active_tab'),
             Input('evaluation-data', 'data'),
             Input('response-data', 'data'),
             Input('comparison-data', 'data')]
        )
        def render_tab_content(active_tab, eval_data, response_data, comparison_data):
            if active_tab == "overview-tab":
                return self.create_overview_content(eval_data)
            elif active_tab == "response-tab":
                return self.create_response_analysis_content(response_data)
            elif active_tab == "comparison-tab":
                return self.create_head_to_head_content(comparison_data)
            elif active_tab == "metrics-tab":
                return self.create_detailed_metrics_content(eval_data)
            elif active_tab == "consistency-tab":
                return self.create_consistency_content(eval_data)
            return html.Div("Select a tab to view content")
        
        @self.app.callback(
            [Output('evaluation-data', 'data'),
             Output('response-data', 'data'),
             Output('comparison-data', 'data'),
             Output('status-alert', 'children'),
             Output('status-alert', 'color'),
             Output('progress-bar', 'value'),
             Output('generate-report-btn', 'disabled')],
            [Input('run-eval-btn', 'n_clicks')],
            [State('model-selector', 'value'),
             State('eval-method', 'value'),
             State('category-selector', 'value')],
            prevent_initial_call=True
        )
        def run_evaluation(n_clicks, models, method, categories):
            if not models:
                return None, None, None, "Please select at least one model", "warning", 0, True
            
            # This would actually run the evaluation
            # Placeholder for demonstration
            return (
                [{'model': m, 'category': 'test', 'overall_score': np.random.random()} 
                 for m in models],
                [{'prompt': 'Test prompt', 'category': 'test', 
                  'model_responses': [{'model': m, 'response': 'Test response', 
                                      'scores': {'overall': np.random.random()}} 
                                     for m in models]}],
                {'comparisons': []},
                f"Successfully evaluated {len(models)} models",
                "success",
                100,
                False
            )
    
    def create_detailed_metrics_content(self, data):
        """Create detailed metrics tab content"""
        if not data:
            return html.Div("No data available")
        
        return html.Div([
            html.H4("Detailed Metrics Analysis"),
            # Add more detailed visualizations here
        ])
    
    def create_consistency_content(self, data):
        """Create consistency analysis tab content"""
        if not data:
            return html.Div("No data available")
        
        return html.Div([
            html.H4("Response Consistency Analysis"),
            html.P("Analyzing how consistent each model is across multiple runs"),
            # Add consistency visualizations here
        ])
    
    def run(self, debug=True, port=8050):
        self.app.run(debug=debug, port=port)