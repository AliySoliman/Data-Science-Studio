import streamlit as st
import pandas as pd
import uuid

def show_visualization_creator(current_data):
    """Show UI for creating new visualizations"""
    st.subheader("➕ Create New Visualization")
    
    # Get globally transformed data
    from visualizations import apply_global_transformations
    global_data = apply_global_transformations()
    
    st.info("💡 Using globally transformed data. Individual transformations can be added below.")
    
    # Data transformations selection for this visualization (additional to global)
    st.subheader("📊 Additional Individual Transformations")
    available_transformations = ["original"] + [step["name"] for step in st.session_state.pipeline['transformations']]
    individual_transformations = st.multiselect(
        "Select additional transformations (applied after global transformations):",
        options=available_transformations,
        default=["original"],
        key="viz_creator_individual_transformations"
    )
    
    # Apply individual transformations to get final preview data
    preview_data = apply_individual_transformations(global_data, individual_transformations)
    
    # Show data preview
    with st.expander("📋 Final Data Preview (global + individual transformations)"):
        st.dataframe(preview_data.head(10))
        st.write(f"Shape: {preview_data.shape}")
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Select Visualization Type:",
        ["Simple Charts", "Advanced Charts", "Statistical Plots"],
        key="viz_type_selector"
    )
    
    if viz_type == "Simple Charts":
        show_simple_charts_ui(current_data, preview_data, individual_transformations)
    elif viz_type == "Advanced Charts":
        show_advanced_charts_ui(current_data, preview_data, individual_transformations)
    elif viz_type == "Statistical Plots":
        show_statistical_plots_ui(current_data, preview_data, individual_transformations)

# ADD: Helper function for individual transformations
def apply_individual_transformations(base_data, transformations):
    """Apply individual transformations on top of global data"""
    if not transformations or ("original" in transformations and len(transformations) == 1):
        return base_data.copy()
    
    df = base_data.copy()
    
    for step_name in transformations:
        if step_name == "original":
            continue
            
        # Find the transformation step
        step = next((s for s in st.session_state.pipeline['transformations'] if s["name"] == step_name), None)
        if step:
            try:
                from transformations import transformation_execution
                df = transformation_execution.execute_transformation(
                    step["type"], 
                    "execution", 
                    {"df": df, "step": step}
                )
            except Exception as e:
                st.error(f"Error applying individual transformation '{step_name}': {str(e)}")
                continue
    
    return df

def show_simple_charts_ui(current_data, preview_data, transformations):
    """UI for simple charts"""
    chart_type = st.selectbox(
        "Chart Type:",
        ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Pie Chart"],
        key="simple_chart_type"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # X-axis selection from preview data
        x_axis = st.selectbox("X-axis:", preview_data.columns, key="x_axis_simple")
        
        # Y-axis selection (for some charts)
        if chart_type in ["Scatter Plot", "Line Chart", "Bar Chart"]:
            y_axis = st.selectbox("Y-axis:", preview_data.columns, key="y_axis_simple")
        else:
            y_axis = None
    
    with col2:
        # Color encoding
        color_by = st.selectbox(
            "Color by (optional):", 
            ["None"] + list(preview_data.columns),
            key="color_by_simple"
        )
        color_by = None if color_by == "None" else color_by
        
        # Size encoding (for scatter plots)
        if chart_type == "Scatter Plot":
            numeric_cols = preview_data.select_dtypes(include='number').columns.tolist()
            size_by = st.selectbox(
                "Size by (optional):",
                ["None"] + numeric_cols,
                key="size_by_simple"
            )
            size_by = None if size_by == "None" else size_by
        else:
            size_by = None
    
    # Advanced options
    with st.expander("Advanced Options"):
        col3, col4 = st.columns(2)
        with col3:
            width = st.slider("Width", 400, 1200, 600, key="width_simple")
            height = st.slider("Height", 300, 800, 400, key="height_simple")
            
            # ADD BINS OPTION FOR HISTOGRAM
            if chart_type == "Histogram":
                bins = st.slider("Number of bins", 5, 100, 30, key="bins_simple")
            else:
                bins = 30  # Default value
                
        with col4:
            title = st.text_input("Chart Title", f"{chart_type}", key="title_simple")
            show_grid = st.checkbox("Show Grid", True, key="grid_simple")
    
    if st.button("Add to Dashboard", key="add_simple"):
        from viz_lib import visualization_execution
        viz_config = {
            "type": "simple",
            "chart_type": chart_type,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "color_by": color_by,
            "size_by": size_by,
            "width": width,
            "height": height,
            "title": title,
            "show_grid": show_grid,
            "transformations": transformations
        }
        
        # ADD BINS TO CONFIG FOR HISTOGRAM
        if chart_type == "Histogram":
            viz_config["bins"] = bins
        
        visualization_execution.create_visualization(current_data, viz_config)

def show_advanced_charts_ui(current_data):
    """UI for advanced charts"""
    chart_type = st.selectbox(
        "Chart Type:",
        ["Heatmap", "3D Scatter", "Parallel Coordinates", "Treemap", "Sunburst", "Violin Plot"],
        key="advanced_chart_type"
    )
    
    if chart_type == "Heatmap":
        show_heatmap_ui(current_data)
    elif chart_type == "3D Scatter":
        show_3d_scatter_ui(current_data)
    elif chart_type == "Parallel Coordinates":
        show_parallel_coords_ui(current_data)
    else:
        st.info(f"{chart_type} coming soon...")

def show_heatmap_ui(current_data, preview_data, transformations):
    """UI for heatmap configuration"""
    numeric_cols = preview_data.select_dtypes(include='number').columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis:", preview_data.columns, key="heatmap_x")
        color_scale = st.selectbox(
            "Color Scale:",
            ["Viridis", "Plasma", "Inferno", "Blues", "Reds", "Greens"],
            key="heatmap_colors"
        )
    with col2:
        y_axis = st.selectbox("Y-axis:", preview_data.columns, key="heatmap_y")
        z_axis = st.selectbox("Values:", numeric_cols, key="heatmap_z")
    
    # Advanced options
    with st.expander("Advanced Options"):
        col3, col4 = st.columns(2)
        with col3:
            width = st.slider("Width", 400, 1200, 600, key="width_heatmap")
            height = st.slider("Height", 300, 800, 400, key="height_heatmap")
        with col4:
            title = st.text_input("Chart Title", "Heatmap", key="title_heatmap")
            show_annotations = st.checkbox("Show Values", True, key="annot_heatmap")
    
    if st.button("Add to Dashboard", key="add_heatmap"):
        from viz_lib import visualization_execution
        viz_config = {
            "type": "advanced",
            "chart_type": "Heatmap",
            "x_axis": x_axis,
            "y_axis": y_axis,
            "z_axis": z_axis,
            "color_scale": color_scale,
            "width": width,
            "height": height,
            "title": title,
            "show_annotations": show_annotations,
            "transformations": transformations  # ADD TRANSFORMATIONS TO CONFIG
        }
        
        visualization_execution.create_visualization(current_data, viz_config)

# Update other advanced chart functions similarly...
def show_3d_scatter_ui(current_data, preview_data, transformations):
    """UI for 3D scatter plot"""
    numeric_cols = preview_data.select_dtypes(include='number').columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis:", numeric_cols, key="3d_x")
        y_axis = st.selectbox("Y-axis:", numeric_cols, key="3d_y")
    with col2:
        z_axis = st.selectbox("Z-axis:", numeric_cols, key="3d_z")
        color_by = st.selectbox(
            "Color by:",
            ["None"] + list(preview_data.columns),
            key="3d_color"
        )
        color_by = None if color_by == "None" else color_by
    
    if st.button("Add to Dashboard", key="add_3d"):
        from viz_lib import visualization_execution
        viz_config = {
            "type": "advanced",
            "chart_type": "3D Scatter",
            "x_axis": x_axis,
            "y_axis": y_axis,
            "z_axis": z_axis,
            "color_by": color_by,
            "width": 800,
            "height": 600,
            "transformations": transformations  # ADD TRANSFORMATIONS TO CONFIG
        }
        
        visualization_execution.create_visualization(current_data, viz_config)

def show_parallel_coords_ui(current_data, preview_data, transformations):
    """UI for parallel coordinates plot"""
    numeric_cols = preview_data.select_dtypes(include='number').columns.tolist()
    
    dimensions = st.multiselect(
        "Select dimensions for parallel coordinates:",
        numeric_cols,
        default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols,
        key="parallel_dims"
    )
    
    color_by = st.selectbox(
        "Color by:",
        ["None"] + list(preview_data.columns),
        key="parallel_color"
    )
    color_by = None if color_by == "None" else color_by
    
    if st.button("Add to Dashboard", key="add_parallel"):
        from viz_lib import visualization_execution
        viz_config = {
            "type": "advanced",
            "chart_type": "Parallel Coordinates",
            "dimensions": dimensions,
            "color_by": color_by,
            "width": 800,
            "height": 500,
            "transformations": transformations  # ADD TRANSFORMATIONS TO CONFIG
        }
        
        visualization_execution.create_visualization(current_data, viz_config)

def show_statistical_plots_ui(current_data):
    """UI for statistical plots"""
    st.info("Statistical plots coming soon...")

def show_dashboard_controls():
    """Show controls for dashboard management"""
    st.subheader("🎛️ Dashboard Controls")
    
    # Show current state
    st.write(f"**Current Visualizations:** {len(st.session_state.visualizations)}")
    
    # Layout options - Initialize if not exists
    if 'dashboard_layout_select' not in st.session_state:
        st.session_state.dashboard_layout_select = "Single Column"
    
    # Get current layout safely
    current_layout = st.session_state.dashboard_layout_select
    layout_options = ["Single Column", "Two Columns", "Three Columns", "Grid"]
    
    # Find current index safely
    try:
        current_index = layout_options.index(current_layout)
    except ValueError:
        current_index = 0  # Default to first option if not found
    
    layout = st.selectbox(
        "Layout:",
        layout_options,
        index=current_index,
        key="dashboard_layout_selector"
    )
    
    # Update session state only if changed
    if layout != st.session_state.dashboard_layout_select:
        st.session_state.dashboard_layout_select = layout
        st.rerun()
    
    # Visualization management
    if st.session_state.visualizations:
        st.write("**Manage Visualizations:**")
        for i, viz in enumerate(st.session_state.visualizations):
            col1, col2, col3 = st.columns([3, 1, 1])  # Changed to 3 columns
            with col1:
                st.write(f"📊 {viz['config'].get('title', f'Viz {i+1}')}")
            with col2:
                if st.button("✏️", key=f"edit_{i}"):  # ADDED EDIT BUTTON
                    edit_visualization(i)
            with col3:
                if st.button("❌", key=f"remove_{i}"):
                    st.session_state.visualizations.pop(i)
                    st.rerun()
        
        if st.button("Clear All", type="secondary"):
            st.session_state.visualizations = []
            st.rerun()
    else:
        st.info("No visualizations added yet")

# ADD THIS NEW FUNCTION for editing visualizations
def edit_visualization(viz_index):
    """Edit an existing visualization"""
    if 'editing_viz_index' not in st.session_state:
        st.session_state.editing_viz_index = viz_index
    else:
        st.session_state.editing_viz_index = viz_index
    
    # Set flag to show edit modal
    st.session_state.show_edit_modal = True

def show_dashboard(current_data):
    """Display the visualization dashboard"""
    if not st.session_state.visualizations:
        st.info("Add visualizations to see them here")
        return
    
    # Get layout setting
    layout = st.session_state.get("dashboard_layout_select", "Single Column")
    
    if layout == "Single Column":
        cols = 1
    elif layout == "Two Columns":
        cols = 2
    elif layout == "Three Columns":
        cols = 3
    else:
        cols = 2
    
    # Display visualizations in grid
    for i in range(0, len(st.session_state.visualizations), cols):
        columns = st.columns(cols)
        for j in range(cols):
            if i + j < len(st.session_state.visualizations):
                with columns[j]:
                    display_single_visualization(st.session_state.visualizations[i + j], current_data)

def display_single_visualization(viz, current_data):
    """Display a single visualization with controls"""
    from viz_lib import visualization_execution
    
    # Get global transformations
    from visualizations import apply_global_transformations
    global_data = apply_global_transformations()
    
    # Show applied transformations
    global_transformations = st.session_state.global_transformations
    individual_transformations = viz['config'].get('transformations', ['original'])
    
    # Display transformation info
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"**🌐 Global:** {', '.join(global_transformations) if global_transformations else 'None'}")
    with col_info2:
        st.info(f"**📊 Individual:** {', '.join(individual_transformations) if individual_transformations else 'None'}")
    
    # Apply individual transformations to global data to get final data
    final_data = apply_individual_transformations(global_data, individual_transformations)
    
    # Display the plot
    fig = visualization_execution.generate_plot(final_data, viz['config'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Comment section for each visualization
    if 'comments' not in viz:
        viz['comments'] = ""
    
    with st.expander("💬 Add Comment"):
        viz['comments'] = st.text_area(
            "Comments:",
            value=viz['comments'],
            key=f"comment_{viz['id']}",
            height=100
        )
    
    # ADD: Report Button and Download Options
    col1, col2, col3, col4 = st.columns(4)  # Changed to 4 columns
    
    with col1:
        if st.button("📥 PNG", key=f"png_{viz['id']}"):
            from viz_lib import visualization_save
            visualization_save.download_plot_as_png(fig, viz['config'].get('title', 'plot'))
    
    with col2:
        if st.button("📥 SVG", key=f"svg_{viz['id']}"):
            from viz_lib import visualization_save
            visualization_save.download_plot_as_svg(fig, viz['config'].get('title', 'plot'))
    
    with col3:
        if st.button("🔄 Update", key=f"update_{viz['id']}"):
            st.rerun()
    
    # ADD: Report Button
    with col4:
        if st.button("📋 Add to Report", key=f"report_{viz['id']}"):
            # Import and show the reporting UI
            from reporting_lib.visualization_reporting import display_visualization_reporting_ui
            st.session_state.show_viz_dialogue = True
            st.session_state.current_viz_id = viz['id']
            st.rerun()
    
    # ADD: Show reporting UI if this visualization is being configured
    if st.session_state.get('show_viz_dialogue', False) and st.session_state.get('current_viz_id') == viz['id']:
        from reporting_lib.visualization_reporting import display_visualization_reporting_ui
        display_visualization_reporting_ui(viz['id'], viz['config'], viz.get('comments', ''))

def show_edit_modal(current_data):
    """Show modal for editing an existing visualization"""
    if not st.session_state.get('show_edit_modal', False):
        return
    
    viz_index = st.session_state.get('editing_viz_index')
    if viz_index is None or viz_index >= len(st.session_state.visualizations):
        st.session_state.show_edit_modal = False
        return
    
    viz = st.session_state.visualizations[viz_index]
    config = viz['config']
    
    with st.container():
        st.header(f"✏️ Edit Visualization: {config.get('title', 'Untitled')}")
        
        # Get the original chart type and category
        chart_type = config.get('chart_type', 'Scatter Plot')
        viz_type = "Simple Charts"  # Default, you might want to detect this from config
        
        # Reuse the existing UI functions but with edit values
        if chart_type in ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Pie Chart"]:
            show_simple_charts_ui_edit(current_data, config, viz_index)
        elif chart_type in ["Heatmap", "3D Scatter", "Parallel Coordinates"]:
            show_advanced_charts_ui_edit(current_data, config, viz_index)
        else:
            st.warning("Edit functionality not available for this chart type")
        
        # Cancel button
        if st.button("Cancel Edit"):
            st.session_state.show_edit_modal = False
            st.session_state.editing_viz_index = None
            st.rerun()

def show_simple_charts_ui_edit(current_data, edit_config, viz_index):
    """Edit UI for simple charts"""
    chart_type = st.selectbox(
        "Chart Type:",
        ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Pie Chart"],
        index=["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Pie Chart"].index(
            edit_config.get('chart_type', 'Scatter Plot')
        ),
        key="edit_simple_chart_type"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # X-axis selection with current value
        x_axis = st.selectbox(
            "X-axis:", 
            current_data.columns, 
            index=current_data.columns.tolist().index(edit_config.get('x_axis', current_data.columns[0])),
            key="edit_x_axis_simple"
        )
        
        # Y-axis selection (for some charts)
        if chart_type in ["Scatter Plot", "Line Chart", "Bar Chart"]:
            y_axis_default = edit_config.get('y_axis', current_data.columns[1] if len(current_data.columns) > 1 else current_data.columns[0])
            y_axis_index = current_data.columns.tolist().index(y_axis_default) if y_axis_default in current_data.columns else 0
            y_axis = st.selectbox(
                "Y-axis:", 
                current_data.columns, 
                index=y_axis_index,
                key="edit_y_axis_simple"
            )
        else:
            y_axis = None
    
    with col2:
        # Color encoding with current value
        color_options = ["None"] + list(current_data.columns)
        color_default = edit_config.get('color_by', "None")
        color_index = color_options.index(color_default) if color_default in color_options else 0
        color_by = st.selectbox(
            "Color by (optional):", 
            color_options,
            index=color_index,
            key="edit_color_by_simple"
        )
        color_by = None if color_by == "None" else color_by
        
        # Size encoding (for scatter plots)
        if chart_type == "Scatter Plot":
            size_options = ["None"] + list(current_data.select_dtypes(include='number').columns)
            size_default = edit_config.get('size_by', "None")
            size_index = size_options.index(size_default) if size_default in size_options else 0
            size_by = st.selectbox(
                "Size by (optional):",
                size_options,
                index=size_index,
                key="edit_size_by_simple"
            )
            size_by = None if size_by == "None" else size_by
        else:
            size_by = None
    
    # Advanced options with current values
    with st.expander("Advanced Options"):
        col3, col4 = st.columns(2)
        with col3:
            width = st.slider("Width", 400, 1200, edit_config.get('width', 600), key="edit_width_simple")
            height = st.slider("Height", 300, 800, edit_config.get('height', 400), key="edit_height_simple")
            
            # ADD BINS FOR HISTOGRAM IN EDIT
            if chart_type == "Histogram":
                bins = st.slider("Number of bins", 5, 100, edit_config.get('bins', 30), key="edit_bins_simple")
            else:
                bins = edit_config.get('bins', 30)
                
        with col4:
            title = st.text_input("Chart Title", edit_config.get('title', f"{chart_type}"), key="edit_title_simple")
            show_grid = st.checkbox("Show Grid", edit_config.get('show_grid', True), key="edit_grid_simple")
    
    # Update button
    if st.button("Update Visualization", key="update_simple"):
        from viz_lib import visualization_execution
        
        updated_config = {
            "type": "simple",
            "chart_type": chart_type,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "color_by": color_by,
            "size_by": size_by,
            "width": width,
            "height": height,
            "title": title,
            "show_grid": show_grid,
            "transformations": edit_config.get('transformations', ['original'])
        }
        
        # ADD BINS FOR HISTOGRAM
        if chart_type == "Histogram":
            updated_config["bins"] = bins
        
        # Update the existing visualization
        st.session_state.visualizations[viz_index]['config'] = updated_config
        
        st.success(f"Visualization '{title}' updated successfully!")
        st.session_state.show_edit_modal = False
        st.session_state.editing_viz_index = None
        st.rerun()

def show_advanced_charts_ui_edit(current_data, edit_config, viz_index):
    """Edit UI for advanced charts"""
    chart_type = edit_config.get('chart_type', 'Heatmap')
    
    if chart_type == "Heatmap":
        show_heatmap_ui_edit(current_data, edit_config, viz_index)
    elif chart_type == "3D Scatter":
        show_3d_scatter_ui_edit(current_data, edit_config, viz_index)
    elif chart_type == "Parallel Coordinates":
        show_parallel_coords_ui_edit(current_data, edit_config, viz_index)
    else:
        st.warning(f"Edit for {chart_type} not implemented yet")

def show_heatmap_ui_edit(current_data, edit_config, viz_index):
    """Edit UI for heatmap configuration"""
    numeric_cols = current_data.select_dtypes(include='number').columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox(
            "X-axis:", 
            current_data.columns, 
            index=current_data.columns.tolist().index(edit_config.get('x_axis', current_data.columns[0])),
            key="edit_heatmap_x"
        )
        color_scales = ["Viridis", "Plasma", "Inferno", "Blues", "Reds", "Greens"]
        color_scale = st.selectbox(
            "Color Scale:",
            color_scales,
            index=color_scales.index(edit_config.get('color_scale', 'Viridis')),
            key="edit_heatmap_colors"
        )
    with col2:
        y_axis = st.selectbox(
            "Y-axis:", 
            current_data.columns, 
            index=current_data.columns.tolist().index(edit_config.get('y_axis', current_data.columns[1] if len(current_data.columns) > 1 else current_data.columns[0])),
            key="edit_heatmap_y"
        )
        z_axis = st.selectbox(
            "Values:", 
            numeric_cols, 
            index=numeric_cols.index(edit_config.get('z_axis', numeric_cols[0])),
            key="edit_heatmap_z"
        )
    
    # Advanced options
    with st.expander("Advanced Options"):
        col3, col4 = st.columns(2)
        with col3:
            width = st.slider("Width", 400, 1200, edit_config.get('width', 600), key="edit_width_heatmap")
            height = st.slider("Height", 300, 800, edit_config.get('height', 400), key="edit_height_heatmap")
        with col4:
            title = st.text_input("Chart Title", edit_config.get('title', "Heatmap"), key="edit_title_heatmap")
            show_annotations = st.checkbox("Show Values", edit_config.get('show_annotations', True), key="edit_annot_heatmap")
    
    if st.button("Update Visualization", key="update_heatmap"):
        from viz_lib import visualization_execution
        viz_config = {
            "type": "advanced",
            "chart_type": "Heatmap",
            "x_axis": x_axis,
            "y_axis": y_axis,
            "z_axis": z_axis,
            "color_scale": color_scale,
            "width": width,
            "height": height,
            "title": title,
            "show_annotations": show_annotations
        }
        
        st.session_state.visualizations[viz_index]['config'] = viz_config
        st.success(f"Visualization '{title}' updated successfully!")
        st.session_state.show_edit_modal = False
        st.session_state.editing_viz_index = None
        st.rerun()

def apply_selected_transformations(selected_steps):
    """Apply selected transformations to the data"""
    if not selected_steps or "original" in selected_steps and len(selected_steps) == 1:
        return st.session_state.df_original.copy()
    
    df = st.session_state.df_original.copy()
    
    for step_name in selected_steps:
        if step_name == "original":
            continue
            
        # Find the transformation step
        step = next((s for s in st.session_state.pipeline['transformations'] if s["name"] == step_name), None)
        if step:
            try:
                from transformations import transformation_execution
                df = transformation_execution.execute_transformation(
                    step["type"], 
                    "execution", 
                    {"df": df, "step": step}
                )
            except Exception as e:
                st.error(f"Error applying transformation '{step_name}': {str(e)}")
                continue
    
    return df