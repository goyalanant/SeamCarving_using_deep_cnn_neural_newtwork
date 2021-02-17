import Bipartite
import home
import DP
import Greedy
import model
import streamlit as st
PAGES = {
    "Home": home,
    "Greedy": Greedy,
    "DP": DP,
    "Bipartite": Bipartite,
    "Model":  model
}
st.sidebar.title('Navigation')

selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.main()
