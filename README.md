# 🏏 ICC Men's T20 World Cup 2026 - Match Winner Predictor

AI-powered cricket match prediction app for T20 World Cup 2026 (India & Sri Lanka).

## 🌟 Features

- **85.83% Accuracy** - Highly accurate predictions using Random Forest
- **Beautiful UI** - T20 World Cup 2026 official theme
- **Real-time Predictions** - Instant winner prediction with confidence scores
- **Comprehensive Analysis** - Team rankings, form, head-to-head, and more

## 🎯 Model Details

- **Algorithm:** Random Forest Classifier
- **Trees:** 100
- **Training Data:** 480 T20 matches
- **Features:** 18 key features including rankings, form, tech index, venue, pitch type, toss

## 🚀 Live Demo

[Click here to try the app](https://your-app-url.streamlit.app)

## 📊 How It Works

1. Select competing teams
2. Enter team statistics (rankings, form, tech index)
3. Choose venue and match conditions
4. Enter toss details
5. Get AI-powered prediction with confidence score

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **ML Model:** Scikit-learn (Random Forest)
- **Data Processing:** Pandas, NumPy
- **Deployment:** Streamlit Cloud

## 📦 Installation (Local)

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/t20-predictor.git
cd t20-predictor

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app_worldcup.py
```

## 📁 Project Structure

```
t20-predictor/
├── app_worldcup.py           # Main Streamlit app
├── cricket_model.pkl         # Trained ML model
├── label_encoders.pkl        # Label encoders for categorical data
├── feature_columns.pkl       # Feature list
├── logo.webp                 # T20 World Cup logo
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## 🎨 Theme

Official ICC Men's T20 World Cup 2026 color scheme:
- 🟠 Orange (#FF8C00)
- 💜 Purple (#4B0082)
- 🩷 Pink (#FF1493)
- 🔵 Navy Blue (#000033)

## 📈 Model Performance

- Training Accuracy: 98.54%
- Testing Accuracy: 85.83%
- Precision (Team A): 91%
- Recall (Team B): 87%

## 🏆 Credits

Built with ❤️ for cricket fans worldwide

## 📝 License

MIT License - Feel free to use and modify!

---

**Enjoy predicting T20 matches! 🏏**
