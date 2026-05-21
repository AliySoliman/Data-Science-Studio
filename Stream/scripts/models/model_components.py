import streamlit as st
from models import html
def comments_section(edit):
                # Make sure comments exist in session_state
            if 'comments' not in st.session_state :
                 if edit :
                     st.session_state.comments = st.session_state.model_data['comments'] if 'comments' in st.session_state.model_data else ""
                 else : st.session_state.comments =""
            # if edit:
            # st.write(edit)
            # st.write(st.session_state.model_data['comments'])
            # st.session_state.setdefault("comments",st.session_state.model_data['comments']if edit else  "")
            st.session_state.setdefault("editing", False)
            # --- 1) handle new chat input FIRST
            prompt = st.chat_input("Write down your highlights!")
            if prompt:
                st.session_state.comments += prompt + "\n"   # accumulate
                # no need to st.rerun(); we render below using updated state
            if st.button("Edit"):
                st.session_state.editing = True
            if st.session_state.comments !="":
                if st.button("save your highlights!!"):
                    add_comments(model_name=st.session_state.model_data['model name'],comments=st.session_state.comments)
                    st.success('comments added successfully')
            # --- 2) UI: edit vs view
            if st.session_state.editing:
                edited_text = st.text_area("Edit your highlights", value=st.session_state.comments, height=200)
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Save Changes"):
                        comments = edited_text
                        st.session_state.editing = False
                        st.rerun()
                with c2:
                    if st.button("Cancel"):
                        st.session_state.editing = False
                        st.rerun()
            else:
                # render HTML (preserve newlines)
                html_comments = st.session_state.comments.replace("\n", "<br>")
                html_template = html.editor(html_comments)
                # f"""
                # <div class="code-container">
                #     <div class="code-header">
                #         <span>MODEL DESCRIPTION</span>
                #         <span>Linear Regression</span>
                #     </div>
                #     <div>{html_comments or "<em>No highlights yet.</em>"}</div>
                # </div>
                # """
                st.markdown(html_template, unsafe_allow_html=True)


def add_comments(model_name,comments):
            for i,saved_model in enumerate(st.session_state.pipeline['ML']) :
                if saved_model['model name'] == model_name :
                    st.session_state.pipeline['ML'][i]['comments'] = comments
                    st.write(st.session_state.pipeline['ML'][i]['comments'])

def model_report():
    results = st.session_state.model_results
    st.markdown("""
                        <div class="report-container">
                            <h3>Model Performance Report</h3>
                            <div class="metric-card">
                                <strong>Features:</strong> {features}<br>
                                <strong>Target:</strong> {target}
                            </div>
                            <div class="metric-card">
                                <strong>Mean Squared Error:</strong> {mse:.4f}<br>
                                <strong>R² Score:</strong> {r2:.4f}
                            </div>
                            <div class="metric-card">
                                <strong>Coefficients:</strong><br>
                                {coeffs}
                            </div>
                            <div class="metric-card">
                                <strong>Intercept:</strong> {intercept:.4f}
                            </div>
                        </div>
                        """.format(
                            features=", ".join(results['features']),
                            target=results['target'],
                            mse=results['metrics']['MSE'],
                            r2=results['metrics']['R2 Score'],
                            coeffs="<br>".join([f"&nbsp;&nbsp;{feat}: {coef:.4f}" 
                                            for feat, coef in zip(results['features'], results['metrics']['Coefficients'])]),
                            intercept=results['metrics']['Intercept']
                        ), unsafe_allow_html=True)
                        # st.write(st.session_state.model_data)