import streamlit as st
import pandas as pd
import asyncio
from openai import AsyncOpenAI
import random
import json
import os
from datetime import datetime, timedelta
from helper_functions import get_stock_price_range, save_to_csv, plot_predictions_over_time

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
            Basierend auf diesen Schlusskursen: {prices}, sage den Schlusskurs für den nächsten Tag voraus. Antworte NUR mit dem vorhergesagten Preis zwischen den Tags <prediction></prediction>.
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
async def make_predictions(csv_file):
    df = pd.read_csv(csv_file, encoding="utf-8")
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    
    client = AsyncOpenAI(api_key=API_KEY)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    valid_indices = list(range(DATA_RANGE, len(df) - 1))
    selected_indices = random.sample(valid_indices, NUM_PREDICTIONS)
    
    tasks = []
    for start_idx in selected_indices:
        historical_slice = df.iloc[start_idx - DATA_RANGE:start_idx]
        historical_data = historical_slice['Close'].tolist()
        prediction_date = df.iloc[start_idx]['Date']
        
        task = asyncio.create_task(
            get_prediction(client, historical_data, prediction_date, semaphore)
        )
        tasks.append((task, start_idx))
    
    results = []
    for task, start_idx in tasks:
        predicted_price, prediction_date = await task
        if predicted_price is not None:
            actual_price = df.iloc[start_idx]['Close']
            last_historical_price = df.iloc[start_idx - 1]['Close']
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
tickers_data = load_tickers_from_json("Data/ticker_jamaican.json")

# Create options for the dropdown
options = [f"{entry['ticker']} - {entry['name']}" for entry in tickers_data]

# Titel der App ändern
st.set_page_config(page_title="TOBANDER Stock App", page_icon="🇯🇲")

# Sidebar with Ticker dropdown
st.sidebar.title("Choose Company 🇯🇲")
selected_option = st.sidebar.selectbox("Select a Ticker", options)

# Extract the selected ticker
ticker = selected_option.split(" - ")[0]

# Get the selected ticker's long name
long_name = next((entry['name'] for entry in tickers_data if entry['ticker'] == ticker), "N/A")

# Zeitraum der letzten 12 Monate
end_date = datetime.today()
start_date = end_date - timedelta(days=365)
st.sidebar.info(f"Zeitraum: {start_date.strftime('%d.%m.%Y')} bis {end_date.strftime('%d.%m.%Y')}")

st.title(f"Predictions for {long_name}")

# Button to load company summary and predictions in the sidebar
if st.sidebar.button("Fetch Company Summary and Run Predictions"):
    # Fetch company summary
    company_summary = asyncio.run(get_company_summary(long_name))
    st.write(company_summary)
    
    # Fetch historical stock price data
    hist_data, long_name = get_stock_price_range(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    if hist_data is not None and long_name:
        # Save data to CSV
        csv_file = f"Data/{ticker}_stock_prices.csv"
        save_to_csv(hist_data, csv_file, long_name)
        st.success(f"Data fetch an save fi {long_name}!")
        
        # Run predictions
        results = asyncio.run(run_predictions(csv_file))
        
        if results:
            # DataFrame to Display the results
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values(by='date', ascending=False)
            
            # Calculate MAE (Mean Absolute Error)
            results_df['absolute_error'] = (results_df['predicted'] - results_df['actual']).abs()
            mae = results_df['absolute_error'].mean()
            
            # Count how many directions were correct
            correct_count = results_df['direction_correct'].sum()
            
            # Format date as DD-MM-YYYY
            results_df['date'] = pd.to_datetime(results_df['date']).dt.strftime('%d-%m-%Y')
            
            # Format numeric columns as USD
            results_df['predicted'] = results_df['predicted'].apply(lambda x: f"${x:,.2f}")
            results_df['actual'] = results_df['actual'].apply(lambda x: f"${x:,.2f}")
            results_df['absolute_error'] = results_df['absolute_error'].apply(lambda x: f"${x:,.2f}")
            
            # Display MAE and No. correct directions
            st.subheader("Performance Metrics")
            st.write(f"MAE: ${mae:,.4f}   #No. Correct Directions: {correct_count}({NUM_PREDICTIONS})")
            
            st.subheader("Prediction Results")
            
            # Spalten DataFrame umbenennen
            results_df = results_df.rename(columns={
                'date': 'Datum',
                'predicted': 'Prediction',
                'actual': 'Actual',
                'direction_correct': 'Correct Direction',
                'absolute_error': 'Absoluter Fehler'
            })
            st.dataframe(results_df[['Datum', 'Prediction', 'Actual', 'Correct Direction', 'Absoluter Fehler']])
            
            # Time Series Plot
            st.subheader("Actual vs. Predicted Kurse")
            fig = plot_predictions_over_time(results_df)
            st.pyplot(fig)
          
        else:
            st.write("Keine Ergebnisse.")
    else:
        st.error("Fehler beim Abholen der Daten.")
