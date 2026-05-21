import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import uuid

def create_visualization(current_data, config):
    """Create and add a new visualization to the dashboard"""
    viz_id = str(uuid.uuid4())[:8]
    
    visualization = {
        "id": viz_id,
        "config": config,
        "comments": ""
    }
    
    if 'visualizations' not in st.session_state:
        st.session_state.visualizations = []
    
    st.session_state.visualizations.append(visualization)
    st.success(f"Visualization '{config.get('title', 'Untitled')}' added to dashboard!")

def generate_plot(current_data, config):
    """Generate plot based on configuration"""
    chart_type = config['chart_type']
    
    # Get global transformations
    from visualizations import apply_global_transformations
    global_data = apply_global_transformations()
    
    # Apply individual transformations if specified
    individual_transformations = config.get('transformations', ['original'])
    plot_data = global_data
    
    if individual_transformations and not (len(individual_transformations) == 1 and individual_transformations[0] == 'original'):
        # Import the helper function
        from viz_lib.visualization_ui import apply_individual_transformations
        plot_data = apply_individual_transformations(global_data, individual_transformations)
    
    try:
        if chart_type == "Scatter Plot":
            fig = create_scatter_plot(plot_data, config)
        elif chart_type == "Line Chart":
            fig = create_line_chart(plot_data, config)
        elif chart_type == "Bar Chart":
            fig = create_bar_chart(plot_data, config)
        elif chart_type == "Histogram":
            fig = create_histogram(plot_data, config)
        elif chart_type == "Box Plot":
            fig = create_box_plot(plot_data, config)
        elif chart_type == "Pie Chart":
            fig = create_pie_chart(plot_data, config)
        elif chart_type == "Heatmap":
            fig = create_heatmap(plot_data, config)
        elif chart_type == "3D Scatter":
            fig = create_3d_scatter(plot_data, config)
        elif chart_type == "Parallel Coordinates":
            fig = create_parallel_coords(plot_data, config)
        else:
            fig = create_default_plot(plot_data, config)
        
        # Apply consistent theme for all plots
        from viz_lib.visualization_save import apply_report_theme
        fig = apply_report_theme(fig)
        
        return fig
        
    except Exception as e:
        st.error(f"Error generating plot: {str(e)}")
        return create_error_plot(str(e))

def create_scatter_plot(data, config):
    """Create scatter plot"""
    fig = px.scatter(
        data,
        x=config['x_axis'],
        y=config['y_axis'],
        color=config.get('color_by'),
        size=config.get('size_by'),
        title=config.get('title', 'Scatter Plot'),
        width=config.get('width', 600),  # USE WIDTH FROM ADVANCED OPTIONS
        height=config.get('height', 400)  # USE HEIGHT FROM ADVANCED OPTIONS
    )
    
    # USE SHOW_GRID FROM ADVANCED OPTIONS
    if config.get('show_grid', True):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    fig.update_traces(marker=dict(opacity=0.7))
    return fig

def create_line_chart(data, config):
    """Create line chart"""
    fig = px.line(
        data,
        x=config['x_axis'],
        y=config['y_axis'],
        color=config.get('color_by'),
        title=config.get('title', 'Line Chart'),
        width=config.get('width', 600),   # USE WIDTH
        height=config.get('height', 400)  # USE HEIGHT
    )
    
    # USE SHOW_GRID
    if config.get('show_grid', True):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    return fig

def create_bar_chart(data, config):
    """Create bar chart"""
    fig = px.bar(
        data,
        x=config['x_axis'],
        y=config['y_axis'],
        color=config.get('color_by'),
        title=config.get('title', 'Bar Chart'),
        width=config.get('width', 600),   # USE WIDTH
        height=config.get('height', 400)  # USE HEIGHT
    )
    
    return fig

def create_histogram(data, config):
    """Create histogram"""
    fig = px.histogram(
        data,
        x=config['x_axis'],
        color=config.get('color_by'),
        title=config.get('title', 'Histogram'),
        width=config.get('width', 600),   # USE WIDTH
        height=config.get('height', 400), # USE HEIGHT
        nbins=config.get('bins', 30)      # USE BINS IF AVAILABLE
    )
    
    return fig

def create_box_plot(data, config):
    """Create box plot"""
    fig = px.box(
        data,
        x=config.get('color_by', config['x_axis']),
        y=config['x_axis'],
        color=config.get('color_by'),
        title=config.get('title', 'Box Plot'),
        width=config.get('width', 600),   # USE WIDTH
        height=config.get('height', 400)  # USE HEIGHT
    )
    
    return fig

def create_pie_chart(data, config):
    """Create pie chart"""
    fig = px.pie(
        data,
        names=config['x_axis'],
        values=config.get('y_axis', data[config['x_axis']].value_counts().index[0]),
        title=config.get('title', 'Pie Chart'),
        width=config.get('width', 600),   # USE WIDTH
        height=config.get('height', 400)  # USE HEIGHT
    )
    
    return fig

def create_heatmap(data, config):
    """Create heatmap"""
    # Pivot data for heatmap
    pivot_data = data.pivot_table(
        values=config['z_axis'],
        index=config['y_axis'],
        columns=config['x_axis'],
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale=config.get('color_scale', 'Viridis'),
        hoverongaps=False,
        text=pivot_data.values if config.get('show_annotations', True) else None,
        texttemplate="%{text:.2f}" if config.get('show_annotations', True) else None
    ))
    
    fig.update_layout(
        title=config.get('title', 'Heatmap'),
        width=config.get('width', 600),   # USE WIDTH
        height=config.get('height', 400), # USE HEIGHT
        xaxis_title=config['x_axis'],
        yaxis_title=config['y_axis']
    )
    
    return fig

def create_3d_scatter(data, config):
    """Create 3D scatter plot"""
    fig = px.scatter_3d(
        data,
        x=config['x_axis'],
        y=config['y_axis'],
        z=config['z_axis'],
        color=config.get('color_by'),
        title=config.get('title', '3D Scatter Plot'),
        width=config.get('width', 800),   # USE WIDTH (larger default for 3D)
        height=config.get('height', 600)  # USE HEIGHT
    )
    
    return fig

def create_parallel_coords(data, config):
    """Create parallel coordinates plot"""
    fig = px.parallel_coordinates(
        data,
        dimensions=config['dimensions'],
        color=config.get('color_by'),
        title=config.get('title', 'Parallel Coordinates Plot'),
        width=config.get('width', 800),   # USE WIDTH
        height=config.get('height', 500)  # USE HEIGHT
    )
    
    return fig

def create_default_plot(data, config):
    """Create a default plot when type is not recognized"""
    fig = go.Figure()
    fig.add_annotation(
        text=f"Plot type '{config.get('chart_type')}' not implemented",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(
        width=config.get('width', 400),
        height=config.get('height', 300)
    )
    return fig

def create_error_plot(error_message):
    """Create an error plot"""
    fig = go.Figure()
    fig.add_annotation(
        text=f"Error: {error_message}",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="red")
    )
    fig.update_layout(
        width=400,
        height=300,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig