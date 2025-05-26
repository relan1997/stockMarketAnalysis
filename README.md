# Stock Market Analysis Web Application

A web-based stock market analysis tool built with Python Flask that provides stock data visualization, analysis, and prediction capabilities through an intuitive web interface.

## Features

- **Stock Ticker Search**: Interactive search functionality to find and analyze specific stocks
- **Historical Data Analysis**: Comprehensive analysis of historical stock performance
- **Price Prediction**: Machine learning-based stock price prediction models
- **Interactive Web Interface**: User-friendly HTML templates for easy navigation
- **Data Visualization**: Charts and graphs for better data interpretation
- **CSV Data Management**: Efficient handling of large stock data sets

## Technology Stack

### Backend
- **Python** - Core programming language
- **Flask** - Web application framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms

### Frontend
- **HTML5** - Structure and content
- **CSS3** - Styling and responsive design
- **JavaScript** - Interactive functionality
- **Bootstrap** (likely) - UI components and styling

### Data Storage
- **CSV Files** - Historical stock data storage
- **File-based system** - Simple and efficient data management

## Project Structure

```
stockMarketAnalysis/
├── static/                    # Static files (CSS, JS, images)
│   └── styles.css            # Main stylesheet
├── templates/                # HTML templates
│   ├── analysis.html         # Stock analysis page
│   ├── index.html           # Home page with ticker search
│   ├── prediction_form.html # Prediction input form
│   └── prediction_res.html  # Prediction results display
├── all_years_stock_data.csv # Complete historical stock dataset
├── stock_data.csv           # Current/processed stock data
├── app.py                   # Main Flask application
└── .gitignore              # Git ignore file
```

## Prerequisites

Before running this application, ensure you have:

- Python 3.7 or higher
- pip (Python package installer)
- Web browser for accessing the application

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/relan1997/stockMarketAnalysis.git
   cd stockMarketAnalysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install flask pandas numpy scikit-learn matplotlib seaborn
   ```

   Or if there's a requirements.txt file:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the Flask application**
   ```bash
   python app.py
   ```

2. **Access the application**
   Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

The application should now be running and accessible through your web browser.

## Application Pages

### 1. Home Page (`index.html`)
- **Ticker Search Bar**: Search for specific stock symbols
- **Navigation**: Access to different analysis features
- **Recent Activity**: Display of recently analyzed stocks

### 2. Analysis Page (`analysis.html`)
- **Stock Data Visualization**: Charts and graphs of stock performance
- **Technical Indicators**: Moving averages, trend analysis
- **Historical Performance**: Long-term stock behavior analysis
- **Comparative Analysis**: Compare multiple stocks

### 3. Prediction Form (`prediction_form.html`)
- **Stock Selection**: Choose stocks for prediction
- **Time Frame Selection**: Set prediction horizons
- **Model Parameters**: Configure prediction models
- **Input Validation**: Ensure valid parameters

### 4. Prediction Results (`prediction_res.html`)
- **Prediction Charts**: Visual representation of predictions
- **Confidence Intervals**: Uncertainty measures
- **Model Performance**: Accuracy metrics and statistics
- **Download Options**: Export results and charts

## Data Files

### `all_years_stock_data.csv`
Complete historical dataset containing:
- Stock symbols and company information
- Daily price data (Open, High, Low, Close, Volume)
- Extended historical records for comprehensive analysis
- Multiple years of market data

### `stock_data.csv`
Processed dataset for current analysis:
- Cleaned and formatted stock data
- Recent market information
- Data optimized for quick analysis and predictions

## Key Features

### Stock Search Functionality
The ticker search bar allows users to:
- Search for stocks by symbol (e.g., AAPL, GOOGL, MSFT)
- Auto-complete suggestions for valid stock symbols
- Quick access to frequently analyzed stocks
- Error handling for invalid symbols

### Analysis Capabilities
- **Price Movement Analysis**: Track stock price changes over time
- **Volume Analysis**: Understand trading patterns
- **Trend Identification**: Identify bullish/bearish trends
- **Statistical Analysis**: Calculate key financial metrics

### Prediction Models
The application likely implements:
- **Linear Regression**: Simple trend-based predictions
- **Moving Average Models**: Technical analysis predictions
- **Time Series Analysis**: Advanced forecasting methods
- **Machine Learning Models**: More sophisticated prediction algorithms

## Configuration

### Flask Application Settings
In `app.py`, you can configure:
```python
# Debug mode for development
app.debug = True

# Port configuration
app.run(port=5000)

# Host configuration (for external access)
app.run(host='0.0.0.0')
```

### Data Processing Settings
- CSV file paths and loading configurations
- Data cleaning and preprocessing parameters
- Model training parameters and thresholds

## Development

### Adding New Features
1. **New Analysis Types**: Add new HTML templates and corresponding Flask routes
2. **Additional Data Sources**: Extend CSV processing or add API integrations
3. **Enhanced Predictions**: Implement new machine learning models
4. **UI Improvements**: Enhance CSS styling and JavaScript functionality

### Code Structure
```python
# Typical Flask app structure in app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    # Stock analysis logic
    return render_template('analysis.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Prediction logic
    return render_template('prediction_form.html')
```

## Styling and UI

### CSS Structure (`static/styles.css`)
The stylesheet likely includes:
- **Responsive Design**: Mobile-friendly layouts
- **Modern UI Elements**: Clean, professional appearance
- **Chart Styling**: Custom styles for data visualizations
- **Interactive Elements**: Hover effects and transitions

### Design Principles
- Clean, minimalist interface
- Intuitive navigation structure
- Professional color scheme suitable for financial data
- Accessible design with proper contrast ratios

## Troubleshooting

### Common Issues

1. **Flask Application Won't Start**
   - Check Python version compatibility
   - Verify all dependencies are installed
   - Ensure port 5000 isn't already in use

2. **Data Loading Errors**
   - Verify CSV files are in the correct location
   - Check file permissions and accessibility
   - Validate CSV file format and structure

3. **Prediction Errors**
   - Ensure sufficient historical data is available
   - Check for missing values in the dataset
   - Verify model parameters are within valid ranges

4. **Template Rendering Issues**
   - Confirm all HTML templates are in the templates/ folder
   - Check for syntax errors in template files
   - Verify static files are properly linked

### Debug Mode
Enable Flask debug mode for development:
```python
if __name__ == '__main__':
    app.run(debug=True)
```

## Deployment

### Local Development
The application is configured for local development by default.

### Production Deployment
For production deployment, consider:
- Using a production WSGI server (Gunicorn, uWSGI)
- Setting up reverse proxy (Nginx)
- Configuring environment variables
- Implementing proper logging and monitoring

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewAnalysis`)
3. Commit your changes (`git commit -m 'Add new analysis feature'`)
4. Push to the branch (`git push origin feature/NewAnalysis`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 Python style guidelines
- Add comments for complex analysis algorithms
- Test new features thoroughly with sample data
- Update documentation for new features

## Future Enhancements

### Planned Features
- Real-time data integration via APIs
- Advanced charting with interactive visualizations
- Portfolio management capabilities
- User authentication and saved preferences
- Mobile-responsive improvements
- Export functionality for analysis results

### Technical Improvements
- Database integration for better data management
- Caching mechanisms for improved performance
- API endpoints for programmatic access
- Enhanced error handling and logging
- Unit tests for critical functions

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This application is for educational and informational purposes only. It should not be considered as financial advice. Always conduct your own research and consult with qualified financial professionals before making investment decisions.

## Author

**relan1997** - [GitHub Profile](https://github.com/relan1997)

## Acknowledgments

- Flask community for the excellent web framework
- Pandas and NumPy developers for data processing capabilities
- Open-source community for various analysis tools
- Financial data providers for historical stock information
