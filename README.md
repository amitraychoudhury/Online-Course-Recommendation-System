🎓 AI-Powered Course Recommendation System

🚀 A Hybrid Recommendation System built using Machine Learning and deployed with Streamlit, designed to provide personalized course recommendations based on user preferences, behavior, and course features.

📌 Project Overview

This project aims to enhance the online learning experience by recommending relevant courses to users using a combination of:

📊 Content-Based Filtering (course similarity)
👥 Collaborative Filtering (user behavior)
🔀 Hybrid Approach (best of both worlds)

The system also includes:
✔️ User Authentication
✔️ Feedback System
✔️ Analytics Dashboard
✔️ PDF Report Generation

⚡ Key Features
🔐 Authentication System
Secure login/signup with hashed passwords
Role-based access (User / Admin)
🤖 Smart Recommendation Engine
Content-Based Filtering (TF-IDF + KNN)
Collaborative Filtering (User KNN)
Hybrid Model with adjustable alpha parameter
🖥️ Interactive UI (Streamlit)
Multi-page app: Home, Recommender, Analytics, Feedback, Admin
Clean UI with course cards (image, rating, difficulty)
📊 Analytics Dashboard
Course popularity insights
Instructor performance
User statistics
📝 Feedback System
Users can rate recommendations
Stored in SQLite database
Admin can monitor feedback
📄 PDF Report Generation
Download personalized recommendations as PDF
🧠 Tech Stack
Category	Tools
Programming	Python 🐍
ML Libraries	Scikit-learn, Pandas, NumPy
NLP	TF-IDF Vectorizer
App Framework	Streamlit
Database	SQLite
Visualization	Matplotlib, Seaborn
Deployment	Local / Streamlit
🏗️ Project Architecture
User → Login System → Recommendation Engine → Results Display
                      ↓
               Feedback System → Database
                      ↓
               Analytics Dashboard
📂 Project Structure
├── app.py
├── saved_models/
│   ├── tfidf_vectorizer.pkl
│   ├── content_knn_model.pkl
│   ├── user_knn_model.pkl
│   ├── df_fe.pkl
│   ├── user_item_matrix.pkl
│
├── assets/
│   ├── course_images/
│   ├── animations/
│
├── database/
│   ├── auth.db
│   ├── feedback.db
│
├── temp/
├── requirements.txt
└── README.md
🚀 How to Run the Project
1️⃣ Clone Repository
git clone https://github.com/your-username/course-recommender.git
cd course-recommender
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run Streamlit App
streamlit run app.py
🔑 Default Login Credentials
Role	Username	Password
Admin	admin	admin123

(You can create new users from Admin panel)

📈 Machine Learning Workflow
Data Cleaning & Preprocessing
Feature Engineering (engagement score, rating category, encoding)
Model Building:
TF-IDF + KNN (Content-Based)
User KNN (Collaborative)
Hybrid Recommendation Logic
Evaluation & Optimization
🎯 Use Cases
📚 Online Learning Platforms (EdTech)
🎓 Course Recommendation Systems
📊 Personalized Content Delivery
🧠 AI-based Learning Assistants
🔮 Future Enhancements
🤖 Deep Learning Models (Neural CF, Transformers)
⚡ Real-time recommendations
🌍 Context-aware personalization
☁️ Cloud deployment (AWS/GCP)
📱 Mobile app integration
👥 Team Members
Sanika S. Sharma
Amit Kumar Raychoudhury
Middi Yogananda Reddy
CHARAN S M
Manjunath B. Chikkabasur
Varsha D V

💡 Conclusion

This project demonstrates how combining Machine Learning + Real-world Application (Streamlit) can create a powerful recommendation system that improves user engagement and decision-making.
