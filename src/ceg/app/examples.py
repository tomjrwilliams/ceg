import streamlit as st
import numpy as np

from ceg.app.nav import Page

class ExamplePage(Page):
    def run(self):
        gen = np.random.default_rng(69)
        vs = gen.normal(0, 1, size = (100,))
        st.line_chart(vs)