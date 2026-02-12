#!/usr/bin/env python3
"""
Test script to verify imports and basic functionality
"""

try:
    import streamlit as st
    print("✅ Streamlit import successful")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")

try:
    import pandas as pd
    print("✅ Pandas import successful")
except ImportError as e:
    print(f"❌ Pandas import failed: {e}")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    print("✅ Plotly imports successful")
    
    # Test a simple plotly function
    import pandas as pd
    test_df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    fig = px.bar(test_df, x='x', y='y')
    print("✅ Plotly Express basic functionality working")
    
except ImportError as e:
    print(f"❌ Plotly import failed: {e}")
except Exception as e:
    print(f"❌ Plotly basic function failed: {e}")

try:
    import numpy as np
    print("✅ NumPy import successful")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

print("\n🔍 Python environment check:")
import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")