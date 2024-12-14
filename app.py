import streamlit as st
import pandas as pd
import asyncio
import nest_asyncio
from openai import AsyncOpenAI
import random
import json
import os
from datetime import datetime, timedelta
import altair as alt
from helper_functions import get_stock_price_range_from_json, plot_predictions_over_time

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Wrapper function for asyncio tasks
def run_asyncio_task(task):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(task)

# ----------------------------------------------------------
# KONSTANTEN UND KONFIGURATION
# ----------------------------------------------------------
DATA_RANGE = 30  
DAYS_AHEAD = 1   
NUM_PREDICTIONS = 30
MAX_CONCURRENT_REQUESTS = 100  
API_KEY = os.getenv('OPENAI_API_KEY') 

# ----------------------------------------------------------
# FUNKTION: EINZELNE VORHERSAGE
# ----------------------------------------------------------
async def get_prediction(client, prices, date, semaphore):
    async with semaphore:
        try:
            prompt = f"""
            Basierend auf diesen Schlusskursen: {prices}, sage den Schlusskurs für den nächsten Tag voraus. 
            Antworte NUR mit dem vorhergesagten Preis zwischen den Tags <prediction></prediction>.
            """
            completion = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Du bist ein Assistent zur Vorhersage von Aktienkursen."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            response = completion.choices[0].message.content
            if "<prediction>" not in response or "</prediction>" not in response:
                return None, date
            prediction_str = response.split("<prediction>")[1].split("</prediction>")[0].strip()
            prediction = float(prediction_str)
            return prediction, date
        except:
            return None, date

# ----------------------------------------------------------
# FUNKTION: MEHRERE VORHERSAGEN ERSTELLEN
# ----------------------------------------------------------
async def make_predictions(json_file):
    df = pd.read_json(json_file, encoding="utf-8")
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    client = AsyncOpenAI(api_key=API_KEY)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    valid_indices = list(range(DATA_RANGE, len(df) - 1))
    selected_indices = random.sample(valid_indices, NUM_PREDICTIONS)
    
    tasks = []
    for start_idx in selected_indices:
        historical_slice = df.iloc[start_idx - DATA_RANGE:start_idx]
        historical_data = historical_slice['Closing Price ($)'].tolist()
        prediction_date = df.iloc[start_idx]['date']
        
        task = asyncio.create_task(
            get_prediction(client, historical_data, prediction_date, semaphore)
        )
        tasks.append((task, start_idx))
    
    results = []
    for task, start_idx in tasks:
        predicted_price, prediction_date = await task
        if predicted_price is not None:
            actual_price = df.iloc[start_idx]['Closing Price ($)']
            last_historical_price = df.iloc[start_idx - 1]['Closing Price ($)']
            predicted_direction = predicted_price - last_historical_price
            actual_direction = actual_price - last_historical_price
            direction_correct = ((predicted_direction > 0 and actual_direction > 0) or
                                 (predicted_direction < 0 and actual_direction < 0))
            
            result = {
                'date': prediction_date,
                'predicted': predicted_price,
                'actual': actual_price,
                'direction_correct': direction_correct
            }
            results.append(result)
    return results

# ----------------------------------------------------------
# HAUPTFUNKTION
# ----------------------------------------------------------
async def run_predictions(csv_file):
    results = await make_predictions(csv_file)
    return results

# ----------------------------------------------------------
# ZUSATZFUNKTION: KURZE UNTERNEHMENS-ZUSAMMENFASSUNG
# ----------------------------------------------------------
async def get_company_summary(company_name):
    client = AsyncOpenAI(api_key=API_KEY)
    prompt = f"Bitte erstelle eine kurze Zusammenfassung des Unternehmens {company_name} in ein bis zwei Sätzen. Auf PATOIS bitte."
    completion = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Du bist ein Assistent für Finanzinformationen."},
            {"role": "user", "content": prompt}
        ]
    )
    response = completion.choices[0].message.content.strip()
    return response

# ----------------------------------------------------------
# STREAMLIT APP
# ----------------------------------------------------------

# Sidebar mit Ticker Dropdown
def load_tickers_from_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        st.error(f"Error loading tickers: {str(e)}")
        return []

# Load Tickers von JSON-File
tickers_data = load_tickers_from_json("Data/ticker_data.json")

# Create options for the dropdown
options = [f"{entry['ticker']} - {entry['long_name']}" for entry in tickers_data]

# Titel der App ändern
st.set_page_config(page_title="TOBANDER Stock App", page_icon="🇯🇲")

# Sidebar with Ticker dropdown
st.sidebar.title("Choose Company 🇯🇲")
selected_option = st.sidebar.selectbox("Select a Ticker", options)

# Extract the selected ticker
ticker = selected_option.split(" - ")[0]

# Get the selected ticker's long name
long_name = next((entry['long_name'] for entry in tickers_data if entry['ticker'] == ticker), "N/A")

# Zeitraum der letzten 12 Monate
end_date = datetime.today()
start_date = end_date - timedelta(days=366)
st.sidebar.info(f"Zeitraum: {start_date.strftime('%d.%m.%Y')} bis {end_date.strftime('%d.%m.%Y')}")

st.title(f"Predictions for {long_name}")

# Button to load company summary and predictions in the sidebar
if st.sidebar.button("Fetch Company Summary and Run Predictions"):
    # Fetch company summary
    company_summary = run_asyncio_task(get_company_summary(long_name))
    st.write(company_summary)
    
    # Fetch historical stock price data
    hist_data, ticker_or_error = get_stock_price_range_from_json(ticker, start_date, end_date)
    
    if hist_data is not None:
        
        # Make Predictions using JSON file
        csv_file = f"Json/{ticker}_data.json"
        st.success(f"Data loaded successfully for {ticker_or_error}!")
        
        # HISTORICAL KURSE BERECHNEN
        latest_price = hist_data.iloc[0]['Closing Price ($)'] # aktueller Kurs
        yesterday_price = hist_data.iloc[1]['Closing Price ($)'] if len(hist_data) > 1 else "N/A" # gestriger Kurs
        last_year_date = hist_data.iloc[0]['date'] - timedelta(days=366) # Datum vor 1 Jahr
        last_year_price_row = hist_data[hist_data['date'] == last_year_date] # Zeile vor 1 Jahr
        last_year_price = last_year_price_row['Closing Price ($)'].iloc[0] if not last_year_price_row.empty else "N/A" # Kurs vor 1 Jahr
        todays_diff = latest_price - yesterday_price
        arrow = "⬆️" if todays_diff > 0 else "⬇️" if todays_diff < 0 else "➖"
        
        # Run predictions
        results = run_asyncio_task(run_predictions(csv_file))
        
        if results:
            # DataFrame to Display the results
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(by='date', ascending=False)
            
            # Calculate Absolute Error
            results_df['absolute_error'] = (results_df['predicted'] - results_df['actual']).abs()
            
            # Count how many directions were correct
            correct_count = results_df['direction_correct'].sum()
            
            # Format date as DD-MM-YYYY
            results_df['date'] = pd.to_datetime(results_df['date']).dt.strftime('%d-%m-%Y')
            
            # Format numeric columns as USD
            results_df['predicted'] = results_df['predicted'].apply(lambda x: f"${x:,.2f}")
            results_df['actual'] = results_df['actual'].apply(lambda x: f"${x:,.2f}")
            results_df['absolute_error'] = results_df['absolute_error'].apply(lambda x: f"${x:,.2f}")
            
            # Display recent Stock Prices
            latest_date = hist_data['date'].max()
            day_before = latest_date - timedelta(days=1)
            year_before = latest_date - timedelta(days=366)
            
            close_max = hist_data['Closing Price ($)'].max()
            close_min = hist_data['Closing Price ($)'].min()
            
            st.subheader("Stock Price Overview")
            st.write(f"📅 **Current Stock Price ({latest_date.strftime('%Y-%m-%d')}):** {latest_price:.2f} $ ({arrow}{todays_diff:.2f})")
            st.write(f"🔙 **Yesterday's Stock Price ({day_before.strftime('%Y-%m-%d')}):** {yesterday_price:.2f} $")
            st.write(f"⏮️ **Stock Price Last Year ({year_before.strftime('%Y-%m-%d')}):** {last_year_price:.2f} $")
            
            # Display Line Chart
            line_chart = alt.Chart(hist_data).mark_line().encode(
                x=alt.X('date:T', title='', axis=alt.Axis(format='%Y-%m')),  # Format x-axis labels
                y=alt.Y('Closing Price ($):Q', title='Closing Price ($)', scale=alt.Scale(domain=[close_min, close_max])),  # Set y-axis limits here
                tooltip=['date:T', 'Closing Price ($):Q']  # Add tooltip for interactivity
            ).properties(
                title="Closing Price",
                width=700,
                height=400
            )

            # Display the Altair chart in Streamlit
            st.altair_chart(line_chart, use_container_width=True)
            
            # Display Prediction Table
            st.subheader(f"Prediction Results {correct_count}({NUM_PREDICTIONS})")
            results_df = results_df.rename(columns={
                'date': 'Datum',
                'predicted': 'Prediction',
                'actual': 'Actual',
                'direction_correct': 'Correct Direction',
                'absolute_error': 'Absoluter Fehler'
            })
            st.dataframe(results_df[['Datum', 'Prediction', 'Actual', 'Correct Direction', 'Absoluter Fehler']])
            
            # Display Time Series Plot
            st.subheader("Actual vs. Predicted Kurse")
            fig = plot_predictions_over_time(results_df)
            st.pyplot(fig)
          
        else:
            st.write("Keine Ergebnisse.")
    else:
        st.error(f"Error: {ticker_or_error}")
