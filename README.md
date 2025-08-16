# 🚀 AI-Powered Resume & Job Match Optimizer

A comprehensive, production-ready Streamlit application that helps job seekers optimize their resumes for Applicant Tracking Systems (ATS) and improve job matching using advanced AI and NLP techniques.

## ✨ Features

### Core Functionality
- **📄 Multi-format Resume Upload**: Support for PDF, DOCX, and TXT files
- **🎯 Advanced ATS Scoring**: Comprehensive scoring algorithm with detailed breakdown
- **🤖 AI-Powered Resume Rewriting**: OpenRouter API integration for intelligent optimization
- **💼 Job Match Analysis**: Semantic similarity and keyword matching
- **🔍 Keyword Gap Analysis**: Identify missing skills and keywords
- **📝 AI Cover Letter Generation**: Personalized cover letters for each application
- **❓ Interview Question Generation**: Prepare for interviews with AI-generated questions

### Advanced Features
- **📊 Analytics Dashboard**: Track ATS score trends and progress over time
- **📋 Multiple Job Comparison**: Compare resume against multiple job descriptions
- **📄 Resume Version Management**: Save and compare different resume versions
- **🔍 Section-by-Section Analysis**: Detailed feedback for each resume section
- **🎯 One-Click Apply Readiness**: Comprehensive application preparation checklist
- **🏭 Industry-Specific Templates**: Optimized templates for Tech, Finance, Marketing
- **📱 Mobile-Responsive Design**: Works seamlessly on all devices

### UI/UX Features
- **🎨 Modern Glassmorphism Design**: Beautiful, modern interface
- **🌙 Dark/Light Mode Toggle**: Customizable theme preferences
- **📊 Interactive Charts**: Plotly-powered visualizations
- **🔄 Real-time Feedback**: Live ATS score updates as you edit
- **📥 Multiple Export Options**: Download optimized resumes and reports

## 🛠️ Technology Stack

- **Backend**: Python, Streamlit
- **AI/NLP**: OpenRouter API, spaCy, sentence-transformers
- **Data Science**: scikit-learn, pandas, numpy
- **Visualization**: Plotly, matplotlib, seaborn
- **Document Processing**: pdfplumber, PyMuPDF, python-docx, docx2txt
- **Database**: SQLite (with optional Firebase support)
- **Authentication**: streamlit-authenticator

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenRouter API key (for AI features)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/resume-optimizer.git
cd resume-optimizer
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download spaCy model**
```bash
python -m spacy download en_core_web_sm
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. **Run the application**
```bash
streamlit run app.py
```

6. **Open your browser**
Navigate to `http://localhost:8501`

## 🔧 Configuration

### API Keys
1. **OpenRouter API**: Required for AI features
   - Sign up at [OpenRouter](https://openrouter.ai/)
   - Add your API key to `.env` file

2. **Firebase (Optional)**: For user authentication
   - Create a Firebase project
   - Add configuration to `.env` file

### Environment Variables
See `.env.example` for all available configuration options.

## 📖 Usage Guide

### 1. Upload Documents
- Navigate to the **Upload** page
- Upload your resume (PDF, DOCX, or TXT)
- Paste or upload the job description

### 2. Analyze Resume
- Go to the **Analysis** page
- View comprehensive ATS scoring
- Review keyword analysis and job matching
- Check detailed recommendations

### 3. Optimize Resume
- Visit the **Optimization** page
- Use AI to rewrite your resume
- Generate personalized cover letters
- Get section-by-section feedback
- Prepare for interviews

### 4. Compare and Track
- Use **Comparison** page for multiple jobs
- Track progress in **Analytics**
- Download reports from **Reports** page

## 🎯 ATS Scoring Algorithm

The application uses a sophisticated scoring system:

- **Keyword Matching (40%)**: TF-IDF based keyword overlap
- **Skills Matching (30%)**: Technical and soft skills alignment
- **Semantic Similarity (20%)**: Sentence transformer embeddings
- **Format & Structure (10%)**: Resume formatting analysis

## 📊 Features Overview

| Feature | Description | Status |
|---------|-------------|--------|
| Resume Upload | Multi-format file support | ✅ Complete |
| ATS Scoring | Advanced scoring algorithm | ✅ Complete |
| AI Optimization | OpenRouter API integration | ✅ Complete |
| Job Matching | Semantic similarity analysis | ✅ Complete |
| Analytics | Progress tracking dashboard | ✅ Complete |
| Export Options | PDF/DOCX downloads | ✅ Complete |
| Mobile Support | Responsive design | ✅ Complete |

## 🔒 Privacy & Security

- **Local Processing**: Most analysis happens locally
- **No Data Storage**: Optional data persistence
- **API Security**: Secure API key management
- **User Consent**: Transparent data usage policies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check this README and code comments
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions

## 🚀 Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Add secrets for API keys
4. Deploy!

### Docker
```bash
docker build -t resume-optimizer .
docker run -p 8501:8501 resume-optimizer
```

### Heroku
```bash
heroku create your-app-name
git push heroku main
```

## 🔮 Roadmap

- [ ] Advanced ML models for better scoring
- [ ] Integration with job boards (LinkedIn, Indeed)
- [ ] Salary insights and market analysis
- [ ] Team collaboration features
- [ ] API endpoints for third-party integrations
- [ ] Advanced resume templates
- [ ] Video interview preparation

## 📈 Performance

- **Fast Processing**: Optimized algorithms for quick analysis
- **Scalable**: Designed for high concurrent usage
- **Efficient**: Minimal resource requirements
- **Cached**: Smart caching for better performance

---

**Built with ❤️ using Streamlit | © 2024**
