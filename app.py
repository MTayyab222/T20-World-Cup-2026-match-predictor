"""
🏏 ICC MEN'S T20 WORLD CUP 2026 - MATCH PREDICTOR
==================================================
India & Sri Lanka 2026
Official Match Winner Prediction App

Theme: T20 World Cup 2026 Official Colors
"""

import streamlit as st
import pandas as pd
import pickle
import base64

# ============================================================================
# CUSTOM CSS - T20 WORLD CUP THEME
# ============================================================================
def local_css():
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&display=swap');
        
        /* Main Background - Navy Blue */
        .stApp {
            background: linear-gradient(135deg, #000033 0%, #1a0052 100%);
            font-family: 'Poppins', sans-serif;
        }
        
        /* Header Styling */
        h1 {
            color: #FF8C00 !important;
            font-family: 'Poppins', sans-serif !important;
            font-weight: 800 !important;
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            padding: 20px;
            background: linear-gradient(90deg, #FF1493, #FF8C00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        h2 {
            color: #FF8C00 !important;
            font-family: 'Poppins', sans-serif !important;
            font-weight: 700 !important;
        }
        
        h3 {
            color: #FF1493 !important;
            font-family: 'Poppins', sans-serif !important;
            font-weight: 600 !important;
        }
        
        /* Cards/Boxes */
        .stTextInput > div > div {
            background-color: rgba(255, 255, 255, 0.1);
            border: 2px solid #FF8C00;
            border-radius: 10px;
        }
        
        .stSelectbox > div > div {
            background-color: rgba(255, 255, 255, 0.1);
            border: 2px solid #FF8C00;
            border-radius: 10px;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #FF8C00 0%, #FF1493 100%);
            color: white;
            font-weight: 700;
            font-size: 18px;
            padding: 15px 30px;
            border-radius: 25px;
            border: none;
            box-shadow: 0 4px 15px rgba(255, 140, 0, 0.4);
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 20, 147, 0.6);
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: #FF8C00 !important;
            font-weight: 800 !important;
            font-size: 32px !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #000033 0%, #1a0052 100%);
            border-right: 3px solid #FF8C00;
        }
        
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #FF8C00 !important;
        }
        
        /* Success/Info Boxes */
        .stSuccess {
            background-color: rgba(255, 140, 0, 0.2);
            border: 2px solid #FF8C00;
            border-radius: 10px;
            padding: 10px;
        }
        
        .stInfo {
            background-color: rgba(255, 20, 147, 0.2);
            border: 2px solid #FF1493;
            border-radius: 10px;
        }
        
        /* Progress Bar */
        .stProgress > div > div {
            background: linear-gradient(90deg, #FF8C00, #FF1493);
        }
        
        /* Divider */
        hr {
            border: 2px solid #FF8C00;
            background: #FF8C00;
        }
        
        /* Text Color */
        p, label, div {
            color: white !important;
        }
        
        /* Winner Announcement Box */
        .winner-box {
            background: linear-gradient(135deg, #FF8C00 0%, #FF1493 100%);
            padding: 30px;
            border-radius: 20px;
            text-align: center;
            box-shadow: 0 8px 30px rgba(255, 140, 0, 0.5);
            margin: 20px 0;
        }
        
        .winner-text {
            color: white;
            font-size: 48px;
            font-weight: 800;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.5);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        /* Confidence Badge */
        .confidence-badge {
            background: linear-gradient(135deg, #4B0082 0%, #FF1493 100%);
            color: white;
            padding: 15px 30px;
            border-radius: 50px;
            font-size: 24px;
            font-weight: 700;
            display: inline-block;
            box-shadow: 0 4px 15px rgba(75, 0, 130, 0.4);
        }
        
        /* Team Cards */
        .team-card {
            background: rgba(255, 255, 255, 0.05);
            border: 2px solid #FF8C00;
            border-radius: 15px;
            padding: 20px;
            margin: 10px 0;
            backdrop-filter: blur(10px);
        }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="T20 World Cup 2026 - Match Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"  # Sidebar always open
)

# Apply custom CSS
local_css()

# ============================================================================
# LOAD MODEL AND ENCODERS
# ============================================================================
@st.cache_resource
def load_model_and_encoders():
    try:
        with open('cricket_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        return model, encoders, feature_columns
    except FileNotFoundError:
        st.error("❌ Required files not found! Make sure .pkl files are in the same folder.")
        st.stop()

model, encoders, feature_columns = load_model_and_encoders()

# Get lists
TEAMS = sorted(encoders['Team_A'].classes_)
VENUES = sorted(encoders['Venue'].classes_)
PITCH_TYPES = sorted(encoders['Pitch_Type'].classes_)
STAGES = sorted(encoders['Stage'].classes_)

# ============================================================================
# HEADER WITH LOGO
# ============================================================================
try:
    col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
    with col_logo2:
        st.image("logo.webp", width=400)
except:
    pass

st.markdown("<h1>🏏 MATCH WINNER PREDICTOR 🏏</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #FF1493;'>INDIA & SRI LANKA 2026</h3>", unsafe_allow_html=True)

# Model Info Box - Always Visible
info_col1, info_col2, info_col3, info_col4 = st.columns(4)
with info_col1:
    st.markdown("""
    <div style='background: rgba(255,140,0,0.2); padding: 15px; border-radius: 10px; border: 2px solid #FF8C00; text-align: center;'>
        <h4 style='color: #FF8C00; margin: 0;'>🎯 Accuracy</h4>
        <h2 style='color: white; margin: 5px 0;'>85.83%</h2>
    </div>
    """, unsafe_allow_html=True)

with info_col2:
    st.markdown("""
    <div style='background: rgba(255,20,147,0.2); padding: 15px; border-radius: 10px; border: 2px solid #FF1493; text-align: center;'>
        <h4 style='color: #FF1493; margin: 0;'>🌲 Algorithm</h4>
        <h2 style='color: white; margin: 5px 0;'>Random Forest</h2>
    </div>
    """, unsafe_allow_html=True)

with info_col3:
    st.markdown("""
    <div style='background: rgba(75,0,130,0.2); padding: 15px; border-radius: 10px; border: 2px solid #4B0082; text-align: center;'>
        <h4 style='color: #4B0082; margin: 0;'>🎓 Trained On</h4>
        <h2 style='color: white; margin: 5px 0;'>480 Matches</h2>
    </div>
    """, unsafe_allow_html=True)

with info_col4:
    st.markdown("""
    <div style='background: rgba(255,140,0,0.2); padding: 15px; border-radius: 10px; border: 2px solid #FF8C00; text-align: center;'>
        <h4 style='color: #FF8C00; margin: 0;'>📊 Trees</h4>
        <h2 style='color: white; margin: 5px 0;'>100</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    # Instruction for users
    st.markdown("""
    <div style='background: linear-gradient(135deg, #FF8C00, #FF1493); 
                padding: 15px; 
                border-radius: 10px; 
                text-align: center;
                margin-bottom: 20px;'>
        <h4 style='color: white; margin: 0;'>👈 MODEL INFO</h4>
        <p style='color: white; margin: 5px 0 0 0; font-size: 12px;'>Expand sidebar to view details</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## 🏆 MODEL INFO")
    
    st.metric("🎯 Accuracy", "85.83%", "Excellent")
    st.metric("🌲 Algorithm", "Random Forest")
    st.metric("📊 Trees", "100")
    st.metric("🎓 Trained On", "480 Matches")
    
    st.markdown("---")
    
    st.markdown("### 🔮 HOW IT WORKS")
    st.markdown("""
    1️⃣ **Select Teams**  
    Choose competing teams
    
    2️⃣ **Enter Stats**  
    Rankings, form, tech index
    
    3️⃣ **Match Details**  
    Venue, pitch, toss info
    
    4️⃣ **Predict**  
    Get AI-powered prediction
    
    5️⃣ **Analyze**  
    View confidence & factors
    """)
    
    st.markdown("---")
    
    st.markdown("### 🎨 POWERED BY")
    st.markdown("""
    - 🤖 Machine Learning
    - 🌲 Random Forest
    - 📊 Scikit-learn
    - 🎨 Streamlit
    """)

st.markdown("---")

# How it Works - Expandable
with st.expander("ℹ️ **HOW DOES THE PREDICTION WORK?** (Click to expand)", expanded=False):
    how_col1, how_col2 = st.columns(2)
    
    with how_col1:
        st.markdown("""
        ### 🔮 Prediction Process
        
        1. **Input Collection**  
           → You provide team details, venue, and match info
        
        2. **Data Encoding**  
           → Text converted to numbers using saved encoders
        
        3. **Feature Engineering**  
           → 18 features prepared for the model
        
        4. **Model Prediction**  
           → Random Forest (100 trees) analyzes patterns
        
        5. **Confidence Calculation**  
           → Probability score for each team
        """)
    
    with how_col2:
        st.markdown("""
        ### 📊 Key Features Used
        
        **Team Strength:**
        - 🏅 World Rankings
        - 📈 Current Form
        - ⚡ Tech Index
        
        **Historical Data:**
        - 🎯 Head-to-Head Record
        - 🏟️ Venue History
        
        **Match Conditions:**
        - 🌾 Pitch Type
        - 🎲 Toss Winner & Decision
        - 🏆 Match Stage
        
        **Model Specs:**
        - Algorithm: Random Forest
        - Trees: 100
        - Accuracy: 85.83%
        - Training Data: 480 matches
        """)

# ============================================================================
# MAIN INPUT FORM
# ============================================================================
st.markdown("## 🎮 MATCH SETUP")

# Team Selection
team_col1, team_col2 = st.columns(2)

with team_col1:
    st.markdown("<div class='team-card'>", unsafe_allow_html=True)
    st.markdown("### 🏏 TEAM A")
    team_a = st.selectbox("Select Team A", TEAMS, key="team_a")
    team_a_ranking = st.number_input("🏅 World Ranking", 1, 20, 5, key="rank_a")
    team_a_form = st.slider("📈 Current Form", 0.0, 100.0, 70.0, 0.1, key="form_a")
    team_a_tech = st.number_input("⚡ Tech Index", 0.0, 400.0, 250.0, 1.0, key="tech_a")
    h2h_a = st.number_input("🎯 H2H Wins", 0, 50, 10, key="h2h_a")
    st.markdown("</div>", unsafe_allow_html=True)

with team_col2:
    st.markdown("<div class='team-card'>", unsafe_allow_html=True)
    st.markdown("### 🏏 TEAM B")
    team_b = st.selectbox("Select Team B", TEAMS, key="team_b")
    team_b_ranking = st.number_input("🏅 World Ranking", 1, 20, 8, key="rank_b")
    team_b_form = st.slider("📈 Current Form", 0.0, 100.0, 65.0, 0.1, key="form_b")
    team_b_tech = st.number_input("⚡ Tech Index", 0.0, 400.0, 200.0, 1.0, key="tech_b")
    h2h_b = st.number_input("🎯 H2H Wins", 0, 50, 8, key="h2h_b")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Match Details
match_col1, match_col2, match_col3 = st.columns(3)

with match_col1:
    st.markdown("### 🏟️ VENUE")
    venue = st.selectbox("Stadium", VENUES)
    pitch = st.selectbox("Pitch Type", PITCH_TYPES)
    avg_score = st.number_input("Avg Score", 100, 250, 165, 5)

with match_col2:
    st.markdown("### 🎲 TOSS")
    toss_winner = st.radio("Toss Winner", ["Team A", "Team B"])
    toss_decision = st.radio("Decision", ["Bat", "Field"])

with match_col3:
    st.markdown("### 🏆 OTHER")
    stage = st.selectbox("Stage", STAGES)
    home_a = st.checkbox("Team A Home")
    home_b = st.checkbox("Team B Home")

# ============================================================================
# PREDICT BUTTON
# ============================================================================
st.markdown("---")

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_btn = st.button("🔮 PREDICT WINNER 🔮", use_container_width=True)

if predict_btn:
    if team_a == team_b:
        st.error("❌ Teams cannot be the same!")
    else:
        with st.spinner("🔮 Analyzing match data..."):
            # Prepare input
            input_data = {
                'Team_A_encoded': encoders['Team_A'].transform([team_a])[0],
                'Team_B_encoded': encoders['Team_B'].transform([team_b])[0],
                'Team_A_Ranking': team_a_ranking,
                'Team_B_Ranking': team_b_ranking,
                'Team_A_Form': team_a_form,
                'Team_B_Form': team_b_form,
                'HeadToHead_A_Wins': h2h_a,
                'HeadToHead_B_Wins': h2h_b,
                'Venue_encoded': encoders['Venue'].transform([venue])[0],
                'Venue_HomeAdvantage_A': 1 if home_a else 0,
                'Venue_HomeAdvantage_B': 1 if home_b else 0,
                'Pitch_Type_encoded': encoders['Pitch_Type'].transform([pitch])[0],
                'Avg_T20_Score_Venue': avg_score,
                'Toss_Winner_Binary': 1 if toss_winner == "Team A" else 0,
                'Toss_Decision_encoded': encoders['Toss_Decision'].transform([toss_decision])[0],
                'Team_A_Tech_Index': team_a_tech,
                'Team_B_Tech_Index': team_b_tech,
                'Stage_encoded': encoders['Stage'].transform([stage])[0]
            }
            
            # Predict
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            
            winner = team_a if prediction == 1 else team_b
            confidence = proba[prediction] * 100
            
            # RESULTS DISPLAY
            st.markdown("---")
            
            # Winner Announcement
            st.markdown(f"""
            <div class='winner-box'>
                <div class='winner-text'>
                    🏆 {winner.upper()} WINS! 🏆
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence
            st.markdown(f"""
            <div style='text-align: center; margin: 30px 0;'>
                <span class='confidence-badge'>
                    Confidence: {confidence:.1f}%
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar
            st.progress(confidence / 100)
            
            st.markdown("---")
            
            # Probabilities
            prob_col1, prob_col2 = st.columns(2)
            
            with prob_col1:
                st.markdown(f"""
                <div style='background: rgba(255,140,0,0.2); padding: 20px; border-radius: 15px; border: 2px solid #FF8C00;'>
                    <h3 style='text-align: center;'>🏏 {team_a}</h3>
                    <h2 style='text-align: center; color: #FF8C00;'>{proba[1]*100:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with prob_col2:
                st.markdown(f"""
                <div style='background: rgba(255,20,147,0.2); padding: 20px; border-radius: 15px; border: 2px solid #FF1493;'>
                    <h3 style='text-align: center;'>🏏 {team_b}</h3>
                    <h2 style='text-align: center; color: #FF1493;'>{proba[0]*100:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Confidence Level
            st.markdown("### 📊 CONFIDENCE ANALYSIS")
            if confidence >= 90:
                st.success("🌟 **VERY HIGH** - Model is extremely confident!")
            elif confidence >= 75:
                st.info("✅ **HIGH** - Strong prediction")
            elif confidence >= 60:
                st.warning("⚠️ **MODERATE** - Close match")
            else:
                st.error("❓ **LOW** - Very close, difficult to predict")
            
            # Key Factors
            st.markdown("---")
            st.markdown("### 🔍 KEY FACTORS")
            
            factor_col1, factor_col2 = st.columns(2)
            
            with factor_col1:
                st.markdown(f"**✨ {team_a} STRENGTHS:**")
                if team_a_ranking < team_b_ranking:
                    st.markdown("✅ Better world ranking")
                if team_a_form > team_b_form:
                    st.markdown("✅ Superior current form")
                if team_a_tech > team_b_tech:
                    st.markdown("✅ Higher tech index")
                if h2h_a > h2h_b:
                    st.markdown("✅ Better H2H record")
            
            with factor_col2:
                st.markdown(f"**✨ {team_b} STRENGTHS:**")
                if team_b_ranking < team_a_ranking:
                    st.markdown("✅ Better world ranking")
                if team_b_form > team_a_form:
                    st.markdown("✅ Superior current form")
                if team_b_tech > team_a_tech:
                    st.markdown("✅ Higher tech index")
                if h2h_b > h2h_a:
                    st.markdown("✅ Better H2H record")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background: rgba(255,140,0,0.1); border-radius: 15px;'>
    <h3 style='color: #FF8C00;'>🏏 ICC MEN'S T20 WORLD CUP 2026 🏏</h3>
    <p style='color: #FF1493; font-size: 18px;'>INDIA & SRI LANKA</p>
    <p style='color: white;'>Match Predictor | Powered by AI & Machine Learning</p>
    <p style='color: #FF8C00; font-weight: 600;'>Model Accuracy: 85.83% | Random Forest (100 Trees)</p>
</div>
""", unsafe_allow_html=True)