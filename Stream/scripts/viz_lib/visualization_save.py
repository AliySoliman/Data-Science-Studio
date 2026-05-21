import streamlit as st
import plotly.io as pio
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import tempfile
import os
import base64
from io import BytesIO
import save  # Import your save module

# Set Plotly default template to ensure colorful exports
pio.templates.default = "plotly"

def check_kaleido_available():
    """Check if kaleido is available and working"""
    try:
        import kaleido
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_annotation(text="Test", x=0.5, y=0.5, showarrow=False)
        pio.write_image(fig, "test.png", engine="kaleido")
        if os.path.exists("test.png"):
            os.remove("test.png")
        return True
    except Exception:
        return False

def apply_export_theme(fig):
    """Apply a light theme with colors for export"""
    # Update layout for better export appearance
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black', size=12),
        title=dict(font=dict(size=16, color='black')),
        xaxis=dict(
            gridcolor='lightgray',
            linecolor='black',
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            gridcolor='lightgray', 
            linecolor='black',
            tickfont=dict(color='black')
        )
    )
    return fig

def show_export_options():
    """Show options for exporting the dashboard"""
    st.subheader("📤 Export Dashboard")
    
    kaleido_available = check_kaleido_available()
    
    if not kaleido_available:
        st.warning("⚠️ Some export features require the 'kaleido' package.")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 Export as PDF", use_container_width=True, disabled=not kaleido_available):
            export_pdf_report()
    
    with col2:
        if st.button("🖼️ Export as PNG", use_container_width=True, disabled=not kaleido_available):
            export_all_as_png()
            
    with col3:
        if st.button("🌐 Export as HTML", use_container_width=True):
            export_as_html_report()
    
    with col4:
        if st.button("💾 Save to Project", use_container_width=True):
            save_dashboard_to_project()

def save_dashboard_to_project():
    """Save dashboard state to the project save file"""
    try:
        # Check if we have a save file path
        if 'save_file_upload' not in st.session_state or not st.session_state.save_file_upload:
            st.error("No save file path configured. Please set a save file path in the Transform page first.")
            return
        
        # Prepare dashboard data for saving
        dashboard_data = {
            "visualizations": st.session_state.visualizations,
            "dashboard_layout": st.session_state.get("dashboard_layout_select", "Single Column"),
            "global_transformations": st.session_state.global_transformations  # ADD GLOBAL TRANSFORMATIONS
        }
        
        # Update the pipeline with dashboard data
        if 'pipeline' not in st.session_state:
            st.session_state.pipeline = {}
        
        st.session_state.pipeline["dashboard"] = dashboard_data
        
        # Use your existing save function
        save.save_encoded_transformations(st.session_state.pipeline)
        
        st.success(f"✅ Dashboard saved successfully to: {st.session_state.save_file_upload}")
        
    except Exception as e:
        st.error(f"Error saving dashboard to project: {str(e)}")

def load_dashboard_from_project():
    """Load dashboard state from the project save file"""
    try:
        # Check if we have a save file path
        if 'save_file_upload' not in st.session_state or not st.session_state.save_file_upload:
            st.error("No save file path configured. Please set a save file path in the Transform page first.")
            return
        
        # Load the pipeline data
        loaded_data = save.load_encoded_transformations(st.session_state.save_file_upload)
        
        if loaded_data and "dashboard" in loaded_data:
            dashboard_data = loaded_data["dashboard"]
            
            # Restore visualizations
            st.session_state.visualizations = dashboard_data.get("visualizations", [])
            
            # Restore layout
            if "dashboard_layout" in dashboard_data:
                st.session_state.dashboard_layout_select = dashboard_data["dashboard_layout"]
            
            # RESTORE GLOBAL TRANSFORMATIONS
            if "global_transformations" in dashboard_data:
                st.session_state.global_transformations = dashboard_data["global_transformations"]
            
            st.success("✅ Dashboard loaded successfully from project file!")
            st.rerun()
        else:
            st.info("No saved dashboard found in the project file.")
            
    except Exception as e:
        st.error(f"Error loading dashboard from project: {str(e)}")

def export_pdf_report():
    """Export the entire dashboard as a PDF report"""
    if not st.session_state.visualizations:
        st.warning("No visualizations to export")
        return
    
    try:
        if not check_kaleido_available():
            st.error("Kaleido is not available for PDF export.")
            return
        
        with tempfile.TemporaryDirectory() as temp_dir:
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
            styles = getSampleStyleSheet()
            story = []
            
            title_style = styles['Heading1']
            title_style.alignment = 1
            story.append(Paragraph("Visualization Dashboard Report", title_style))
            story.append(Spacer(1, 0.3*inch))
            
            successful_exports = 0
            for i, viz in enumerate(st.session_state.visualizations):
                from viz_lib import visualization_execution
                
                current_data = st.session_state.df_original
                fig = visualization_execution.generate_plot(current_data, viz['config'])
                
                # Apply export theme to ensure colors are preserved
                fig = apply_export_theme(fig)
                
                img_path = os.path.join(temp_dir, f"viz_{i}.png")
                try:
                    # Export with high quality and color preservation
                    pio.write_image(
                        fig, 
                        img_path, 
                        width=800, 
                        height=600, 
                        engine="kaleido",
                        scale=2,  # Higher resolution
                        format='png'
                    )
                    successful_exports += 1
                except Exception as e:
                    st.error(f"Failed to export visualization {i+1}: {str(e)}")
                    continue
                
                viz_title = Paragraph(f"Visualization {i+1}: {viz['config'].get('title', 'Untitled')}", styles['Heading2'])
                story.append(viz_title)
                
                try:
                    img = Image(img_path, width=6*inch, height=4.5*inch)
                    story.append(img)
                except Exception as e:
                    st.error(f"Failed to add image {i+1} to PDF: {str(e)}")
                    continue
                
                if viz.get('comments'):
                    story.append(Paragraph("<b>Comments:</b>", styles['Heading3']))
                    story.append(Paragraph(viz['comments'], styles['Normal']))
                
                story.append(Spacer(1, 0.2*inch))
                
                if (i + 1) % 2 == 0 and (i + 1) < len(st.session_state.visualizations):
                    story.append(PageBreak())
            
            if successful_exports == 0:
                st.error("No visualizations could be exported to PDF.")
                return
            
            doc.build(story)
            pdf_data = buffer.getvalue()
            b64 = base64.b64encode(pdf_data).decode()
            href = f'<a href="data:application/pdf;base64,{b64}" download="dashboard_report.pdf">Download PDF Report ({successful_exports} visualizations)</a>'
            st.markdown(href, unsafe_allow_html=True)
            st.success(f"PDF report generated with {successful_exports} visualizations!")
            
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")

def export_all_as_png():
    """Export all visualizations as individual PNG files"""
    if not st.session_state.visualizations:
        st.warning("No visualizations to export")
        return
    
    try:
        if not check_kaleido_available():
            st.error("Kaleido is not available for PNG export.")
            return
            
        from viz_lib import visualization_execution
        current_data = st.session_state.df_original
        
        successful_exports = 0
        for i, viz in enumerate(st.session_state.visualizations):
            fig = visualization_execution.generate_plot(current_data, viz['config'])
            
            # Apply export theme to ensure colors are preserved
            fig = apply_export_theme(fig)
            
            title = viz['config'].get('title', f'viz_{i+1}').replace(' ', '_')
            
            try:
                # Export with high quality settings
                img_bytes = pio.to_image(
                    fig, 
                    format="png", 
                    width=1200, 
                    height=800, 
                    engine="kaleido",
                    scale=2  # Higher resolution for better quality
                )
                b64 = base64.b64encode(img_bytes).decode()
                href = f'<a href="data:image/png;base64,{b64}" download="{title}.png">Download {title}.png</a>'
                st.markdown(href, unsafe_allow_html=True)
                successful_exports += 1
            except Exception as e:
                st.error(f"Failed to export {title}: {str(e)}")
        
        if successful_exports > 0:
            st.success(f"Successfully exported {successful_exports} PNG files!")
            
    except Exception as e:
        st.error(f"Error exporting PNGs: {str(e)}")

def download_plot_as_png(fig, title):
    """Download individual plot as PNG"""
    try:
        if not check_kaleido_available():
            st.error("Kaleido is not available for PNG export. Using HTML fallback.")
            download_plot_as_html(fig, title)
            return
        
        # Apply export theme to ensure colors are preserved
        fig = apply_export_theme(fig)
        
        # Export with high quality settings
        img_bytes = pio.to_image(
            fig, 
            format="png", 
            width=1200, 
            height=800, 
            engine="kaleido",
            scale=2  # Higher resolution
        )
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{title}.png">Download PNG (High Quality)</a>'
        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error downloading PNG: {str(e)}")
        download_plot_as_html(fig, title)

def download_plot_as_svg(fig, title):
    """Download individual plot as SVG"""
    try:
        if not check_kaleido_available():
            st.error("Kaleido is not available for SVG export. Using HTML fallback.")
            download_plot_as_html(fig, title)
            return
        
        # Apply export theme to ensure colors are preserved
        fig = apply_export_theme(fig)
        
        img_bytes = pio.to_image(
            fig, 
            format="svg", 
            width=1200, 
            height=800, 
            engine="kaleido"
        )
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="data:image/svg+xml;base64,{b64}" download="{title}.svg">Download SVG (Vector)</a>'
        st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error downloading SVG: {str(e)}")
        download_plot_as_html(fig, title)

def download_plot_as_html(fig, title):
    """Download plot as HTML (fallback method)"""
    try:
        # Apply export theme for HTML as well
        fig = apply_export_theme(fig)
        
        import plotly.offline as pyo
        html_content = pyo.plot(fig, include_plotlyjs=True, output_type='div')
        b64 = base64.b64encode(html_content.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="{title}.html">Download HTML (Interactive)</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.info("HTML export preserves interactivity (zoom, hover, etc.)")
    except Exception as e:
        st.error(f"Error downloading HTML: {str(e)}")

def export_as_html_report():
    """Export dashboard as HTML report (reliable fallback)"""
    if not st.session_state.visualizations:
        st.warning("No visualizations to export")
        return
    
    try:
        from viz_lib import visualization_execution
        import plotly.offline as pyo
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Visualization Dashboard Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    margin: 20px; 
                    background-color: #ffffff;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                .viz-container { 
                    margin: 30px 0; 
                    border: 1px solid #ddd; 
                    padding: 20px; 
                    border-radius: 8px;
                    background: white;
                }
                .viz-title { 
                    font-size: 20px; 
                    font-weight: bold; 
                    margin-bottom: 15px;
                    color: #2c3e50;
                }
                .viz-comment { 
                    margin-top: 15px; 
                    font-style: italic; 
                    color: #555;
                    padding: 10px;
                    background: #f8f9fa;
                    border-left: 4px solid #3498db;
                }
                h1 {
                    color: #2c3e50;
                    text-align: center;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📊 Visualization Dashboard Report</h1>
        """
        
        current_data = st.session_state.df_original
        
        for i, viz in enumerate(st.session_state.visualizations):
            fig = visualization_execution.generate_plot(current_data, viz['config'])
            
            # Apply export theme for HTML export too
            fig = apply_export_theme(fig)
            
            plot_html = pyo.plot(fig, include_plotlyjs=False, output_type='div')
            
            html_content += f"""
            <div class="viz-container">
                <div class="viz-title">📈 Visualization {i+1}: {viz['config'].get('title', 'Untitled')}</div>
                {plot_html}
            """
            
            if viz.get('comments'):
                html_content += f'<div class="viz-comment">💬 Comments: {viz["comments"]}</div>'
            
            html_content += "</div>"
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        b64 = base64.b64encode(html_content.encode()).decode()
        href = f'<a href="data:text/html;base64,{b64}" download="dashboard_report.html">📥 Download Interactive HTML Report</a>'
        st.markdown(href, unsafe_allow_html=True)
        st.success("✅ HTML report generated successfully!")
        
    except Exception as e:
        st.error(f"Error generating HTML report: {str(e)}")

def apply_report_theme(fig):
    """Apply a consistent theme for report visualizations"""
    # Update layout for better appearance in reports
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black', size=12),
        title=dict(font=dict(size=16, color='black')),
        xaxis=dict(
            gridcolor='lightgray',
            linecolor='black',
            tickfont=dict(color='black')
        ),
        yaxis=dict(
            gridcolor='lightgray', 
            linecolor='black',
            tickfont=dict(color='black')
        ),
        margin=dict(l=50, r=50, t=50, b=50)  # Add some margin
    )
    return fig