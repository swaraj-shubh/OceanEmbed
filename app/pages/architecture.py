import streamlit as st
from pathlib import Path
from PIL import Image

# Setup page config
logo_path = Path(__file__).parent.parent / "logo.png"
if logo_path.exists():
    page_icon = Image.open(logo_path)
else:
    page_icon = "🌊"

st.set_page_config(page_title="OceanEmbed — Architecture", page_icon=page_icon, layout="wide")

# Basic dark theme CSS for the markdown page
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #000000 0%, #0a0a0a 100%);
    background-attachment: fixed;
    color: #e0e0e0;
}
h1, h2, h3, h4 {
    color: #ffffff !important;
    font-weight: 700;
}
pre {
    background: rgba(20, 20, 20, 0.7) !important;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3) !important;
}
th, td {
    background: rgba(20, 20, 20, 0.5);
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
</style>
""", unsafe_allow_html=True)

# Try to load the header logo perfectly aligned like main app
if logo_path.exists():
    import base64
    logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 2rem;">
            <img src="data:image/png;base64,{logo_b64}" width="48" style="background-color: white; border-radius: 50%; padding: 2px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);" />
            <h1 style="margin: 0; padding: 0;">OceanEmbed Architecture</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.title("OceanEmbed Architecture")

# Read and display the markdown content from docs
docs_path = Path(__file__).parent.parent.parent / "docs" / "03-architecture.md"

if docs_path.exists():
    content = docs_path.read_text(encoding="utf-8")
    
    # Strip Jekyll front matter if present
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            content = parts[2].strip()
            
    # Remove the first H1 since we already have a custom title
    lines = content.split('\n')
    cleaned_lines = [line for line in lines if not line.startswith('# 03')]
    content = '\n'.join(cleaned_lines).strip()
    
    st.markdown(content)
else:
    st.error(f"Could not find architecture documentation at `{docs_path.resolve()}`")
