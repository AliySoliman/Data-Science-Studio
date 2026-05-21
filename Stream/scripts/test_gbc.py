import pandas as pd
import numpy as np
import streamlit as st
from models.scripts.Classification.Gradient_Boosting_Classifier.model_script import model_script
from sklearn.datasets import make_classification

df = pd.DataFrame(make_classification(n_samples=100, n_features=4, random_state=42)[0], columns=['A', 'B', 'C', 'D'])
df['target'] = make_classification(n_samples=100, n_features=4, random_state=42)[1]

class DummySessionState(dict):
    def __getattr__(self, key):
        return self.get(key, None)
    def __setattr__(self, key, value):
        self[key] = value

st.session_state = DummySessionState()

try:
    res = model_script(df, ['A', 'B', 'C', 'D'], 'target', False, False, {}, {}, 5)
    print("Result:", "Success" if res else "Failure")
    if res:
        print("Pipeline ML len:", len(st.session_state.pipeline.get('ML', [])))
except Exception as e:
    print("Exception thrown:", e)
