# app.py — Robust single-file Streamlit Recommender (fixed KeyErrors + feedback)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path

# Optional Lottie
try:
    from streamlit_lottie import st_lottie
    import json
except Exception:
    st_lottie = None

# PDF libs
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB = True
except Exception:
    REPORTLAB = False
    try:
        from fpdf import FPDF
        FPDF_OK = True
    except Exception:
        FPDF_OK = False

# --------------------------
# Setup folders & DB
# --------------------------
os.makedirs("saved_models", exist_ok=True)
os.makedirs("assets/course_images", exist_ok=True)
os.makedirs("assets/animations", exist_ok=True)
os.makedirs("database", exist_ok=True)
os.makedirs("temp", exist_ok=True)

AUTH_DB = "database/auth.db"
FB_DB = "database/feedback.db"

# --------------------------
# Simple auth (demo only)
# --------------------------
def init_auth_db():
    conn = sqlite3.connect(AUTH_DB)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT,
            role TEXT
        )
    """)
    conn.commit()
    conn.close()

def hash_password(p): return hashlib.sha256(p.encode()).hexdigest()
def add_user(u,p,role="user"):
    conn = sqlite3.connect(AUTH_DB); cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users(username,password_hash,role) VALUES (?,?,?)",(u,hash_password(p),role))
        conn.commit()
    except sqlite3.IntegrityError:
        pass
    conn.close()

def verify_user(u,p):
    conn = sqlite3.connect(AUTH_DB); cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username=?",(u,))
    r = cur.fetchone(); conn.close()
    return bool(r) and r[0]==hash_password(p)

def get_role(u):
    conn = sqlite3.connect(AUTH_DB); cur = conn.cursor()
    cur.execute("SELECT role FROM users WHERE username=?",(u,))
    r = cur.fetchone(); conn.close()
    return r[0] if r else None

init_auth_db()
add_user("admin","admin123","admin")

# --------------------------
# Feedback DB init & log
# --------------------------
def init_feedback_db():
    conn = sqlite3.connect(FB_DB); cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user TEXT,
        rec_type TEXT,
        course TEXT,
        feedback TEXT,
        rating INTEGER,
        created_at TEXT
    )
    """)
    conn.commit(); conn.close()

def log_feedback(user, rec_type, course, feedback, rating):
    conn = sqlite3.connect(FB_DB); cur = conn.cursor()
    cur.execute("INSERT INTO feedback (user, rec_type, course, feedback, rating, created_at) VALUES (?,?,?,?,?,?)",
                (user, rec_type, course, feedback, int(rating), datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

init_feedback_db()

# --------------------------
# PDF generator (reportlab or fpdf fallback)
# --------------------------
def generate_pdf_report(recs_df, user, file_path="temp/recommendations.pdf"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if REPORTLAB:
        c = canvas.Canvas(file_path, pagesize=letter)
        w,h = letter
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, h-40, f"Recommendations for {user} — {datetime.utcnow().strftime('%Y-%m-%d')}")
        c.setFont("Helvetica", 10)
        y = h-70
        for i, row in recs_df.reset_index(drop=True).iterrows():
            text = f"{i+1}. {row.get('course_name','[N/A]')} | Instructor: {row.get('instructor','[N/A]')} | Rating: {row.get('rating','')}"
            c.drawString(40, y, text[:120])
            y -= 18
            if y < 60:
                c.showPage(); y = h-40
        c.save(); return file_path
    elif FPDF_OK:
        pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial","B",14); pdf.cell(0,10,f"Recommendations for {user}",ln=1)
        pdf.set_font("Arial","",12)
        for i, row in recs_df.reset_index(drop=True).iterrows():
            pdf.multi_cell(0,8,f"{i+1}. {row.get('course_name','[N/A]')} | Instructor: {row.get('instructor','[N/A]')} | Rating: {row.get('rating','')}")
        pdf.output(file_path); return file_path
    else:
        raise RuntimeError("No PDF library available. Install reportlab or fpdf.")

# --------------------------
# Load models / data (defensive)
# --------------------------
@st.cache_data(show_spinner=True)
def load_models():
    M = {}
    M['tfidf'] = joblib.load("saved_models/tfidf_vectorizer.pkl") if os.path.exists("saved_models/tfidf_vectorizer.pkl") else None
    M['content_knn'] = joblib.load("saved_models/content_knn_model.pkl") if os.path.exists("saved_models/content_knn_model.pkl") else None
    M['user_knn'] = joblib.load("saved_models/user_knn_model.pkl") if os.path.exists("saved_models/user_knn_model.pkl") else None
    M['df_fe'] = joblib.load("saved_models/df_fe.pkl") if os.path.exists("saved_models/df_fe.pkl") else pd.DataFrame()
    M['user_item_matrix'] = joblib.load("saved_models/user_item_matrix.pkl") if os.path.exists("saved_models/user_item_matrix.pkl") else pd.DataFrame()
    M['indices'] = joblib.load("saved_models/course_index_map.pkl") if os.path.exists("saved_models/course_index_map.pkl") else pd.Series(dtype=int)
    M['user_index_map'] = joblib.load("saved_models/user_index_map.pkl") if os.path.exists("saved_models/user_index_map.pkl") else []
    return M

models = load_models()
tfidf = models['tfidf']; content_knn = models['content_knn']; user_knn = models['user_knn']
df_fe = models['df_fe']; user_item_matrix = models['user_item_matrix']
indices = models['indices']; user_index_map = models['user_index_map']

# --------------------------
# Defensive column fixes
# --------------------------
# If course_name missing, try to build friendly names from encoded or indices
if isinstance(df_fe, pd.DataFrame):
    if 'course_name' not in df_fe.columns:
        if 'course_name_encoded' in df_fe.columns:
            # create human-readable fallback names
            df_fe['course_name'] = df_fe['course_name_encoded'].apply(lambda x: f"Course_{x}")
        else:
            df_fe['course_name'] = df_fe.index.astype(str).apply(lambda x: f"Course_{x}")

    if 'instructor' not in df_fe.columns:
        if 'instructor_encoded' in df_fe.columns:
            df_fe['instructor'] = df_fe['instructor_encoded'].apply(lambda x: f"Instructor_{x}")
        else:
            df_fe['instructor'] = "Unknown Instructor"

    # ensure course_name_encoded exists when possible
    if 'course_name_encoded' not in df_fe.columns:
        # create from unique course_name map
        df_fe['course_name_encoded'] = pd.factorize(df_fe['course_name'])[0]

# Build decode maps (encoded -> name) for safe use
course_decode_map = dict(zip(df_fe['course_name_encoded'], df_fe['course_name'])) if 'course_name_encoded' in df_fe.columns and 'course_name' in df_fe.columns else {}
instr_decode_map = dict(zip(df_fe.get('instructor_encoded', pd.Series(dtype=int)), df_fe['instructor'])) if 'instructor_encoded' in df_fe.columns else {}

# Ensure indices is a Series mapping course_name -> index; if not provided, attempt to build
if not isinstance(indices, pd.Series) or indices.empty:
    # Build indices mapping from df_fe course_name to df_fe.index
    try:
        indices = pd.Series(df_fe.index.values, index=df_fe['course_name'].values)
    except Exception:
        indices = pd.Series(dtype=int)

# --------------------------
# Basic UI / Styling
# --------------------------
st.set_page_config(page_title="AI Course Recommender", page_icon="🎓", layout="wide")
st.markdown("""
<style>
.stApp { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.header { font-size:32px; font-weight:800; color:#032a3b; margin-bottom:6px; }
.card { background:white; padding:12px; border-radius:10px; box-shadow: 0 6px 18px rgba(0,0,0,0.08); }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Auth UI (sidebar)
# --------------------------
if 'auth' not in st.session_state:
    st.session_state['auth'] = {'logged_in': False, 'username': None, 'role': None}

st.sidebar.header("🔐 Login")
if not st.session_state['auth']['logged_in']:
    lu = st.sidebar.text_input("Username", key="login_user")
    lp = st.sidebar.text_input("Password", type="password", key="login_pass")
    if st.sidebar.button("Login"):
        if verify_user(lu, lp):
            st.session_state['auth'] = {'logged_in': True, 'username': lu, 'role': get_role(lu)}
            st.sidebar.success(f"Welcome {lu} ({st.session_state['auth']['role']})")
        else:
            st.sidebar.error("Invalid credentials")
else:
    st.sidebar.markdown(f"**Logged in:** {st.session_state['auth']['username']} ({st.session_state['auth']['role']})")
    if st.sidebar.button("Logout"):
        st.session_state['auth'] = {'logged_in': False, 'username': None, 'role': None}
        st.experimental_rerun()

# Navigation
page = st.sidebar.selectbox("Navigate", ["Home","Recommender","Analytics","Feedback Dashboard","About","Admin"])

# --------------------------
# Helper: show course card
# --------------------------
def img_path(name):
    p = Path("assets/course_images")/name
    if p.exists(): return str(p)
    return "assets/course_images/default.jpg"

def show_course_card(row):
    st.image(img_path(row.get('thumbnail','default.jpg')), width=220)
    st.markdown(f"**{row.get('course_name','')}**")
    st.markdown(f"👨‍🏫 {row.get('instructor','Unknown')}  •  ⭐ {row.get('rating','')}")
    st.markdown(f"🔖 {row.get('difficulty_level','')}  • {row.get('rating_category','')}")
    st.divider()

# --------------------------
# HOME PAGE
# --------------------------
if page == "Home":
    st.markdown("<div class='header'>🎓 AI Course Recommendation System</div>", unsafe_allow_html=True)
    st.write("Smart course suggestions using hybrid recommender (content + collaborative).")
    if st_lottie:
        anim = None
        try:
            with open("assets/animations/learning.json","r") as f:
                anim = json.load(f)
        except Exception:
            anim = None
        if anim:
            st_lottie(anim, height=200)
    st.markdown("### Project Goal")
    st.markdown("""
    The goal of this dataset is to build an online course recommendation system that suggests relevant courses
    to learners based on their interests, past enrollments, and engagement levels. The dataset includes ratings,
    instructor info, study materials, certification offerings and engagement features — suitable for content-based,
    collaborative or hybrid recommenders.
    """)

# --------------------------
# RECOMMENDER PAGE
# --------------------------
elif page == "Recommender":

    if not st.session_state['auth']['logged_in']:
        st.warning("Please login to use the recommender.")
        st.stop()

    st.header("🤖 Personalized Course Recommender")

    left, right = st.columns([3, 1])

    # --------------------------
    # SIDEBAR CONTROLS
    # --------------------------
    with right:
        algo = st.selectbox(
            "Recommendation Algorithm",
            ["Hybrid", "Content Based", "Collaborative"]
        )
        top_n = st.slider("Top N Recommendations", 3, 12, 5)
        alpha = st.slider("Alpha (Content weight)", 0.0, 1.0, 0.6)

    # --------------------------
    # USER SELECTION (ONLY USER)
    # --------------------------
    with left:
        if 'user_id' in df_fe.columns:
            user_id = st.selectbox(
                "Select User ID",
                sorted(df_fe['user_id'].unique())
            )
        else:
            st.error("User ID not available in dataset.")
            st.stop()

    # =====================================================
    # CONTENT-BASED (AUTO using user's last course)
    # =====================================================
    def content_based_auto(u_id, n=5):

        if tfidf is None or content_knn is None:
            return df_fe.sort_values('rating', ascending=False).head(n)

        user_courses = df_fe[df_fe['user_id'] == u_id]['course_name']

        if user_courses.empty:
            return df_fe.sort_values('rating', ascending=False).head(n)

        last_course = user_courses.iloc[-1]
        key = last_course.lower().strip()

        if key not in indices.index:
            return df_fe.sort_values('rating', ascending=False).head(n)

        idx = indices[key]
        d, idxs = content_knn.kneighbors(
            tfidf.transform([last_course]),
            n_neighbors=min(n + 1, len(df_fe))
        )

        idxs = idxs.flatten()[1:]
        return df_fe.iloc[idxs].copy()

    # =====================================================
    # COLLABORATIVE
    # =====================================================
    def collaborative_auto(u_id, n=5):

        if user_knn is None or user_item_matrix is None or user_item_matrix.empty:
            return df_fe.sort_values('rating', ascending=False).head(n)

        if u_id not in user_item_matrix.index:
            return df_fe.sort_values('rating', ascending=False).head(n)

        uidx = user_index_map.index(u_id)

        d, idxs = user_knn.kneighbors(
            user_item_matrix.iloc[uidx].values.reshape(1, -1),
            n_neighbors=min(6, user_item_matrix.shape[0])
        )

        similar_users = [user_index_map[i] for i in idxs.flatten()[1:]]

        if not similar_users:
            return df_fe.sort_values('rating', ascending=False).head(n)

        sim_scores = (1 - d.flatten()[1:])
        denom = sim_scores.sum() if sim_scores.sum() != 0 else 1

        weighted = user_item_matrix.loc[similar_users].T.dot(sim_scores) / denom
        seen = user_item_matrix.loc[u_id]
        weighted = weighted[seen == 0]

        top_encoded = weighted.sort_values(ascending=False).head(n).index

        return (
            df_fe[df_fe['course_name_encoded'].isin(top_encoded)]
            .drop_duplicates('course_name_encoded')
            .head(n)
            .copy()
        )

    # =====================================================
    # HYBRID (AUTO)
    # =====================================================
    def hybrid_auto(u_id, n=5, alpha=0.6):

        content_df = content_based_auto(u_id, n * 3)
        collab_df = collaborative_auto(u_id, n * 3)

        def names(df):
            if df is None or df.empty:
                return []
            return df['course_name'].tolist()

        c_list = names(content_df)
        col_list = names(collab_df)

        scores = {}

        for i, c in enumerate(c_list):
            scores[c] = scores.get(c, 0) + alpha * (len(c_list) - i)

        for i, c in enumerate(col_list):
            scores[c] = scores.get(c, 0) + (1 - alpha) * (len(col_list) - i)

        if not scores:
            return df_fe.sort_values('rating', ascending=False).head(n)

        final_courses = [
            x[0] for x in sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ][:n]

        return (
            df_fe[df_fe['course_name'].isin(final_courses)]
            .drop_duplicates('course_name')
            .head(n)
            .copy()
        )

    # --------------------------
    # RUN RECOMMENDER
    # --------------------------
    if algo == "Content Based":
        results = content_based_auto(user_id, top_n)
    elif algo == "Collaborative":
        results = collaborative_auto(user_id, top_n)
    else:
        results = hybrid_auto(user_id, top_n, alpha)

    # --------------------------
    # DISPLAY RESULTS
    # --------------------------
    st.markdown("### 🎯 Recommended Courses")

    if results is None or results.empty:
        st.info("No recommendations available.")
    else:
        cols = st.columns(3)
        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % 3]:
                show_course_card(row)

    st.caption("✅ Recommendations are automatically personalized based on user history.")


# --------------------------
# ANALYTICS PAGE
# --------------------------
elif page == "Analytics":
    if not st.session_state['auth']['logged_in']:
        st.warning("Please login.")
        st.stop()
    st.header("📊 Analytics")
    if df_fe.empty:
        st.error("Data not loaded (df_fe empty).")
        st.stop()
    c1,c2,c3 = st.columns(3)
    c1.metric("Courses", df_fe['course_id'].nunique() if 'course_id' in df_fe.columns else len(df_fe))
    c2.metric("Users", df_fe['user_id'].nunique() if 'user_id' in df_fe.columns else 0)
    c3.metric("Avg Rating", round(df_fe['rating'].mean(),2) if 'rating' in df_fe.columns else "N/A")

    st.subheader("Top courses by enrollment")
    if 'enrollment_numbers' in df_fe.columns:
        top_by_enroll = df_fe.groupby(['course_id','course_name'])['enrollment_numbers'].sum().reset_index().sort_values('enrollment_numbers',ascending=False).head(10)
        st.table(top_by_enroll[['course_name','enrollment_numbers']])
    else:
        st.info("No enrollment data available.")

    st.subheader("Instructor Summary")
    # If instructor present, show real summary; otherwise, show instructor_encoded summary or message
    if 'instructor' in df_fe.columns and df_fe['instructor'].notna().any():
        instr = df_fe.groupby('instructor').agg(avg_rating=('rating','mean'), courses=('course_id','nunique')).reset_index().sort_values('avg_rating',ascending=False).head(10)
        st.dataframe(instr)
    elif 'instructor_encoded' in df_fe.columns:
        # map encoded to pseudo-names
        df_fe['instructor_name_tmp'] = df_fe['instructor_encoded'].apply(lambda x: instr_decode_map.get(x, f"Instructor_{x}"))
        instr = df_fe.groupby('instructor_name_tmp').agg(avg_rating=('rating','mean'), courses=('course_id','nunique')).reset_index().sort_values('avg_rating',ascending=False).head(10)
        st.dataframe(instr)
    else:
        st.info("Instructor information not available for analytics.")

# --------------------------
# FEEDBACK DASHBOARD
# --------------------------
elif page == "Feedback Dashboard":
    if not st.session_state['auth']['logged_in']:
        st.warning("Please login.")
        st.stop()
    st.header("⭐ Feedback Dashboard (from SQLite)")
    conn = sqlite3.connect(FB_DB)
    try:
        fb_df = pd.read_sql_query("SELECT * FROM feedback ORDER BY created_at DESC", conn)
    except Exception:
        fb_df = pd.DataFrame()
    conn.close()
    if fb_df.empty:
        st.info("No feedback records found yet. Once users submit feedback it will appear here.")
    else:
        st.dataframe(fb_df.head(500))
        st.markdown("### Aggregates")
        agg = fb_df.groupby('course').agg(avg_rating=('rating','mean'), count=('id','count')).reset_index().sort_values('avg_rating',ascending=False)
        st.dataframe(agg.head(50))

# --------------------------
# ABOUT
# --------------------------
elif page == "About":
    st.header("About & Team")
    st.markdown("""
    **Project Goal**  
    The goal of this dataset is to build an online course recommendation system that suggests relevant courses to learners based on their interests, past enrollments, and engagement levels.
    """)
    st.markdown("### Team")
    st.markdown("""
    **Team Name:** Amit Choudhury  
    **Role:** Data Scientist  
    **Skills:** Python, ML, SQL, Streamlit, Tableau, Excel, ChatGPT  
    **Vision:** Creating AI solutions to solve real business challenges 💡
    """)

# --------------------------
# ADMIN
# --------------------------
elif page == "Admin":
    if not st.session_state['auth']['logged_in'] or st.session_state['auth']['role'] != 'admin':
        st.warning("Admin access required.")
        st.stop()
    st.header("Admin Console")
    st.subheader("Create new user")
    new_user = st.text_input("Username", key="new_user")
    new_pw = st.text_input("Password", type="password", key="new_pw")
    role = st.selectbox("Role", ["user","admin"])
    if st.button("Create user"):
        if new_user and new_pw:
            add_user(new_user,new_pw,role)
            st.success(f"User {new_user} created.")
        else:
            st.error("Provide username and password.")
    st.subheader("Feedback (raw)")
    conn = sqlite3.connect(FB_DB)
    df_fb = pd.read_sql_query("SELECT * FROM feedback ORDER BY created_at DESC", conn)
    conn.close()
    st.dataframe(df_fb.head(500))

# End of app
